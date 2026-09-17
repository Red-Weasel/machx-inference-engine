// tools/glm52_run.cpp — GLM-5.2 (`glm-dsa`) end-to-end forward on one B70.
//
// Runs the real 196.76 GiB model: MLA attention in absorbed form, 256-expert
// top-8 MoE with a shared expert, dense prefix blocks, tied-free lm_head.
// Weights are read straight out of the mmap'd GGUF; Q6_0 tensors are SoA-repacked
// at load time (7.55x on the GEMV, bit-exact — docs/glm52/OPTIMIZATION_LOG.md §2)
// and live in VRAM; the 182 GiB IQ2_KT routed-expert bank stays host-resident and
// is pulled through an LRU VRAM slot cache on demand.
//
// DSA is deliberately NOT implemented. indexer.top_k is 2048, so for any context
// at or below 2048 the sparse path selects every position and dense MLA is
// EXACTLY equivalent — not an approximation. ik_llama.cpp agrees: with -dsa off
// "the model runs the dense MLA path" (build_deepseek2.cpp:802).
// The NextN/MTP block (78) is skipped; it only serves speculative decoding.
//
// Graph verified against ik_llama.cpp src/graphs/build_deepseek2.cpp, which is
// what `glm-dsa` actually dispatches to. RoPE is LLAMA_ROPE_TYPE_NORM (adjacent
// pairs), kq_scale = 1/sqrt(256), routing is sigmoid -> +exp_probs_b for
// SELECTION ONLY -> top-8 -> unbiased probs -> normalise -> x2.5.
//
// usage: ie-glm52-run <model.gguf> [n_predict] [cache_experts] [prompt_tokens...]
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <unistd.h>

using namespace ie;

// Persistent pread pool.
//
// Expert fetches were reading straight out of the mmap'd GGUF and ran at
// 10.8 GB/s against a 26.6 GB/s link, because the driver has to pin and release
// every non-registered source region on every transfer. Copying the whole
// 182 GiB bank into pinned memory fixes the rate but locks almost all of RAM and
// destabilises the machine. This is the bounded version: a small ring of pinned
// staging buffers, filled by parallel pread() from the page cache.
//
// Measured on this box, pread from cache: 1 thread 1.46 GB/s, 2 -> 20.0,
// 4 -> 28.8, 8 -> 29.3, 12 -> 32.0 GB/s. Four threads already clear the PCIe
// link, so the staging copy is not the bottleneck -- and the whole thing costs
// ~80 MiB of pinned memory instead of 182 GiB.
struct PreadPool {
    struct Job { int fd; void* dst; uint64_t off; size_t n; };
    std::vector<std::thread> th;
    std::mutex m;
    std::condition_variable cv, cv_done;
    std::vector<Job> jobs;
    size_t next = 0, done = 0;
    bool stop = false;

    explicit PreadPool(int n) {
        for (int i = 0; i < n; ++i) th.emplace_back([this] { loop(); });
    }
    ~PreadPool() {
        { std::lock_guard<std::mutex> lk(m); stop = true; }
        cv.notify_all();
        for (auto& t : th) t.join();
    }
    void loop() {
        for (;;) {
            Job j;
            {
                std::unique_lock<std::mutex> lk(m);
                cv.wait(lk, [this] { return stop || next < jobs.size(); });
                if (stop) return;
                j = jobs[next++];
            }
            size_t got = 0;
            while (got < j.n) {
                ssize_t r = pread(j.fd, (char*)j.dst + got, j.n - got, off_t(j.off + got));
                if (r <= 0) break;
                got += size_t(r);
            }
            {
                std::lock_guard<std::mutex> lk(m);
                ++done;
            }
            cv_done.notify_one();
        }
    }
    // Run `js` to completion. Callers batch a whole layer's misses so the pool
    // stays saturated instead of being woken once per 3.2 MiB matrix.
    void run(std::vector<Job>& js) {
        if (js.empty()) return;
        {
            std::lock_guard<std::mutex> lk(m);
            jobs.swap(js); next = 0; done = 0;
        }
        cv.notify_all();
        std::unique_lock<std::mutex> lk(m);
        cv_done.wait(lk, [this] { return done == jobs.size(); });
        jobs.clear();
    }
};
using Clock = std::chrono::steady_clock;
static double dsec(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------- quant decode

static constexpr int8_t kIq4kValues[16] = {
    -127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113 };
static constexpr uint32_t KA = 0xCBAC1FED;
static constexpr uint32_t KA_P[8] = {
    KA, KA*KA, KA*KA*KA, KA*KA*KA*KA, KA*KA*KA*KA*KA,
    KA*KA*KA*KA*KA*KA, KA*KA*KA*KA*KA*KA*KA, KA*KA*KA*KA*KA*KA*KA*KA };

// Sum of the four 6-bit lanes, minus 126.
//
// Every byte of `a` is masked to <= 0x3f, so a + (a>>16) puts b0+b2 and b1+b3 in
// bytes 0 and 1 with no carry (each <= 126), and one more fold puts the full sum
// in byte 0 (<= 252). That is 6 ops instead of the 11 of four shift-and-mask
// extracts -- 8 per weight including the multiply and mask, against 13.
//
// It matters most at DECODE, where a routed expert's weight is decoded once and
// used for exactly ONE token: the trellis is then ~92% of the kernel's work.
static inline int trellis_at(uint32_t seed, int j) {
    uint32_t a = (KA_P[j] * seed) & 0x3f3f3f3fu;
    a += a >> 16;
    a += a >> 8;
    return int(a & 0xFFu) - 126;
}

// Q6_0 SoA row layout: [ d : K/32 fp16 ][ qs : K/2 B ][ qh : K/4 B ]
static inline int64_t q6_soa_row_bytes(int64_t K) { return (K/32)*2 + K/2 + K/4; }

static void q6_repack_row(const uint8_t* src, uint8_t* dst, int64_t K) {
    const int64_t nb = K/32;
    uint8_t* pd = dst, *pqs = dst + nb*2, *pqh = pqs + K/2;
    for (int64_t b = 0; b < nb; ++b) {
        const uint8_t* blk = src + b*26;
        std::memcpy(pd + b*2, blk, 2);
        std::memcpy(pqh + b*8, blk + 2, 8);
        std::memcpy(pqs + b*16, blk + 10, 16);
    }
}

// ---------------------------------------------------------------- kernels

// y[n] = w[n] * x[n] / rms(x)
static sycl::event k_rmsnorm(sycl::queue& q, const float* x, const float* w,
                             float* y, int64_t n, float eps) {
    constexpr int WG = 256;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float,1> red(sycl::range<1>(WG/32), h);
        h.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it)
                       [[sycl::reqd_sub_group_size(32)]] {
            const int tid = int(it.get_local_id(0));
            float s = 0.f;
            for (int64_t i = tid; i < n; i += WG) s += x[i]*x[i];
            auto sg = it.get_sub_group();
            s = sycl::reduce_over_group(sg, s, sycl::plus<float>());
            if (sg.get_local_id()[0]==0) red[sg.get_group_id()[0]] = s;
            it.barrier(sycl::access::fence_space::local_space);
            float tot = 0.f;
            for (int i = 0; i < WG/32; ++i) tot += red[i];
            const float inv = sycl::rsqrt(tot/float(n) + eps);
            for (int64_t i = tid; i < n; i += WG) y[i] = x[i]*inv*w[i];
        });
    });
}

// Q6_0 SoA GEMV: y[row] = sum_k W[row][k]*x[k]
static sycl::event k_q6_gemv(sycl::queue& q, const uint8_t* w, const float* x,
                             float* y, int64_t K, int64_t N, int64_t row_bytes,
                             const std::vector<sycl::event>& deps = {}) {
    const int64_t nb = K/32;
    constexpr int WG = 128;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float,1> red(sycl::range<1>(WG/32), h);
        h.parallel_for(sycl::nd_range<1>(N*WG, WG), [=](sycl::nd_item<1> it)
                       [[sycl::reqd_sub_group_size(32)]] {
            const int64_t row = it.get_group(0);
            const int tid = int(it.get_local_id(0));
            const uint8_t* rp = w + row*row_bytes;
            const sycl::half* pd = reinterpret_cast<const sycl::half*>(rp);
            const uint8_t* pqs = rp + nb*2;
            const uint8_t* pqh = pqs + K/2;
            float acc = 0.f;
            for (int64_t b = tid; b < nb; b += WG) {
                const float d = float(pd[b]);
                uint32_t qs[4], qh[2];
                std::memcpy(qs, pqs + b*16, 16);
                std::memcpy(qh, pqh + b*8, 8);
                const float* xv = x + b*32;
                float part = 0.f;
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const uint32_t sb = (qs[j>>2] >> (8*(j&3))) & 0xFFu;
                    const uint32_t hb = (qh[(j>>2)&1] >> (8*(j&3))) & 0xFFu;
                    const int l0 = int(sb & 0x0F), l1 = int(sb >> 4);
                    const int h0 = int((hb >> (4*(j/8))) & 3);
                    const int h1 = int((hb >> (4*(j/8)+2)) & 3);
                    part += float((l0 | (h0<<4)) - 32) * xv[j];
                    part += float((l1 | (h1<<4)) - 32) * xv[j+16];
                }
                acc += d*part;
            }
            auto sg = it.get_sub_group();
            acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            if (sg.get_local_id()[0]==0) red[sg.get_group_id()[0]] = acc;
            it.barrier(sycl::access::fence_space::local_space);
            if (tid==0) { float s=0.f; for (int i=0;i<WG/32;++i) s+=red[i]; y[row]=s; }
        });
    });
}

// Batched Q6_0 SoA GEMV over `nslice` independent (weight-slice, input, output)
// triples. The MLA absorb matrices are stored as 64 per-head slices, so without
// this the forward would launch 64 kernels per matrix per block — 10k launches a
// token, which is pure overhead. One launch instead.
//   slice s: weights at w + s*N*row_bytes, input x + s*x_stride, output y + s*y_stride
static sycl::event k_q6_gemv_multi(sycl::queue& q, const uint8_t* w, const float* x,
                                   int64_t x_stride, float* y, int64_t y_stride,
                                   int64_t K, int64_t N, int64_t row_bytes, int nslice) {
    const int64_t nb = K/32;
    constexpr int WG = 64;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float,1> red(sycl::range<1>(WG/32), h);
        h.parallel_for(sycl::nd_range<1>(int64_t(nslice)*N*WG, WG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int64_t wg = it.get_group(0);
            const int s = int(wg / N);
            const int64_t row = wg % N;
            const int tid = int(it.get_local_id(0));
            const uint8_t* rp = w + (int64_t(s)*N + row)*row_bytes;
            const sycl::half* pd = reinterpret_cast<const sycl::half*>(rp);
            const uint8_t* pqs = rp + nb*2;
            const uint8_t* pqh = pqs + K/2;
            const float* xs = x + int64_t(s)*x_stride;
            float acc = 0.f;
            for (int64_t b = tid; b < nb; b += WG) {
                const float d = float(pd[b]);
                uint32_t qs[4], qh[2];
                std::memcpy(qs, pqs + b*16, 16);
                std::memcpy(qh, pqh + b*8, 8);
                const float* xv = xs + b*32;
                float part = 0.f;
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const uint32_t sb = (qs[j>>2] >> (8*(j&3))) & 0xFFu;
                    const uint32_t hb = (qh[(j>>2)&1] >> (8*(j&3))) & 0xFFu;
                    const int l0 = int(sb & 0x0F), l1 = int(sb >> 4);
                    const int h0 = int((hb >> (4*(j/8))) & 3);
                    const int h1 = int((hb >> (4*(j/8)+2)) & 3);
                    part += float((l0 | (h0<<4)) - 32) * xv[j];
                    part += float((l1 | (h1<<4)) - 32) * xv[j+16];
                }
                acc += d*part;
            }
            auto sg = it.get_sub_group();
            acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            if (sg.get_local_id()[0]==0) red[sg.get_group_id()[0]] = acc;
            it.barrier(sycl::access::fence_space::local_space);
            if (tid==0) { float t=0.f; for (int i=0;i<WG/32;++i) t+=red[i];
                          y[int64_t(s)*y_stride + row]=t; }
        });
    });
}

// Small-K variant of the batched GEMV: ONE WORK-ITEM PER OUTPUT ROW.
// The MLA absorb matrices are narrow — wk_b has K=192 (6 blocks/row) and wv_b
// K=512 (16 blocks/row). A work-group-per-row kernel puts 64 lanes on 6 blocks
// of work and idles 90% of them. Here each lane owns a whole row, so every lane
// is busy and no reduction or barrier is needed at all.
static sycl::event k_q6_gemv_multi_small(sycl::queue& q, const uint8_t* w,
                                         const float* x, int64_t x_stride,
                                         float* y, int64_t y_stride,
                                         int64_t K, int64_t N, int64_t row_bytes,
                                         int nslice) {
    const int64_t nb = K/32;
    constexpr int WG = 256;
    const int64_t total = int64_t(nslice)*N;
    return q.parallel_for(
        sycl::nd_range<1>(((total + WG - 1)/WG)*WG, WG), [=](sycl::nd_item<1> it) {
            const int64_t gid = it.get_global_id(0);
            if (gid >= total) return;
            const int s = int(gid / N);
            const uint8_t* rp = w + gid*row_bytes;
            const sycl::half* pd = reinterpret_cast<const sycl::half*>(rp);
            const uint8_t* pqs = rp + nb*2;
            const uint8_t* pqh = pqs + K/2;
            const float* xs = x + int64_t(s)*x_stride;
            float acc = 0.f;
            for (int64_t b = 0; b < nb; ++b) {
                const float d = float(pd[b]);
                uint32_t qs[4], qh[2];
                std::memcpy(qs, pqs + b*16, 16);
                std::memcpy(qh, pqh + b*8, 8);
                const float* xv = xs + b*32;
                float part = 0.f;
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const uint32_t sb = (qs[j>>2] >> (8*(j&3))) & 0xFFu;
                    const uint32_t hb = (qh[(j>>2)&1] >> (8*(j&3))) & 0xFFu;
                    const int l0 = int(sb & 0x0F), l1 = int(sb >> 4);
                    const int h0 = int((hb >> (4*(j/8))) & 3);
                    const int h1 = int((hb >> (4*(j/8)+2)) & 3);
                    part += float((l0 | (h0<<4)) - 32) * xv[j];
                    part += float((l1 | (h1<<4)) - 32) * xv[j+16];
                }
                acc += d*part;
            }
            const int64_t row = gid % N;
            y[int64_t(s)*y_stride + row] = acc;
        });
}

// Batched Q6_0 SoA GEMM: Y[t][row] = sum_k W[row][k] * X[t][k], for T tokens.
//
// This is the whole prefill story. A GEMV re-reads every weight for every token,
// so a 12-token prompt reads the 13.4 GiB resident set 12 times. Here each
// work-group reads a row's weights ONCE and applies them to all T token vectors,
// so weight traffic is amortised T-fold and the kernel goes from
// memory-bound to arithmetic-bound.
template <int TMAX_, int WG_>
static sycl::event k_q6_gemm_v1t(sycl::queue& q, const uint8_t* w, const float* x,
                             float* y, int64_t K, int64_t N, int64_t row_bytes,
                             int T, const std::vector<sycl::event>& deps) {
    const int64_t nb = K/32;
    constexpr int WG = WG_;
    constexpr int TMAX = TMAX_;              // register tile over tokens
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float,1> red(sycl::range<1>(size_t(WG/32)*TMAX), h);
        h.parallel_for(sycl::nd_range<1>(N*WG, WG), [=](sycl::nd_item<1> it)
                       [[sycl::reqd_sub_group_size(32)]] {
            const int64_t row = it.get_group(0);
            const int tid = int(it.get_local_id(0));
            const uint8_t* rp = w + row*row_bytes;
            const sycl::half* pd = reinterpret_cast<const sycl::half*>(rp);
            const uint8_t* pqs = rp + nb*2;
            const uint8_t* pqh = pqs + K/2;
            // Tokens are processed TMAX at a time: `acc` is a fixed register tile,
            // so running the loop to an arbitrary T wrote past its end (and past
            // the `red` scratch), corrupting local memory for any chunk > 16.
            auto sg = it.get_sub_group();
            const int sgid = int(sg.get_group_id()[0]);
            for (int t0 = 0; t0 < T; t0 += TMAX) {
                const int tn = sycl::min(TMAX, T - t0);
                float acc[TMAX];
                for (int t = 0; t < TMAX; ++t) acc[t] = 0.f;
                for (int64_t b = tid; b < nb; b += WG) {
                    const float d = float(pd[b]);
                    uint32_t qs[4], qh[2];
                    std::memcpy(qs, pqs + b*16, 16);
                    std::memcpy(qh, pqh + b*8, 8);
                    float wv[32];
#pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        const uint32_t sb = (qs[j>>2] >> (8*(j&3))) & 0xFFu;
                        const uint32_t hb = (qh[(j>>2)&1] >> (8*(j&3))) & 0xFFu;
                        const int l0 = int(sb & 0x0F), l1 = int(sb >> 4);
                        const int h0 = int((hb >> (4*(j/8))) & 3);
                        const int h1 = int((hb >> (4*(j/8)+2)) & 3);
                        wv[j]      = d * float((l0 | (h0<<4)) - 32);
                        wv[j + 16] = d * float((l1 | (h1<<4)) - 32);
                    }
                    for (int t = 0; t < tn; ++t) {
                        const float* xv = x + int64_t(t0 + t)*K + b*32;
                        float p = 0.f;
#pragma unroll
                        for (int j = 0; j < 32; ++j) p += wv[j]*xv[j];
                        acc[t] += p;
                    }
                }
                for (int t = 0; t < tn; ++t) {
                    float v = sycl::reduce_over_group(sg, acc[t], sycl::plus<float>());
                    if (sg.get_local_id()[0]==0) red[size_t(sgid)*TMAX + t] = v;
                }
                it.barrier(sycl::access::fence_space::local_space);
                if (tid == 0)
                    for (int t = 0; t < tn; ++t) {
                        float sum = 0.f;
                        for (int i = 0; i < WG/32; ++i) sum += red[size_t(i)*TMAX + t];
                        y[int64_t(t0 + t)*N + row] = sum;
                    }
                it.barrier(sycl::access::fence_space::local_space);
            }
        });
    });
}

static sycl::event k_q6_gemm_v1(sycl::queue& q, const uint8_t* w, const float* x,
                                float* y, int64_t K, int64_t N, int64_t row_bytes,
                                int T, const std::vector<sycl::event>& deps) {
    return k_q6_gemm_v1t<16,128>(q, w, x, y, K, N, row_bytes, T, deps);
}

// Q8_0 GEMV (router + indexer projections). block = fp16 d + 32 int8.
static sycl::event k_q8_gemv(sycl::queue& q, const uint8_t* w, const float* x,
                             float* y, int64_t K, int64_t N) {
    const int64_t nb = K/32, row_bytes = nb*34;
    constexpr int WG = 64;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float,1> red(sycl::range<1>(WG/32), h);
        h.parallel_for(sycl::nd_range<1>(N*WG, WG), [=](sycl::nd_item<1> it)
                       [[sycl::reqd_sub_group_size(32)]] {
            const int64_t row = it.get_group(0);
            const int tid = int(it.get_local_id(0));
            const uint8_t* rp = w + row*row_bytes;
            float acc = 0.f;
            for (int64_t b = tid; b < nb; b += WG) {
                const uint8_t* blk = rp + b*34;
                sycl::half dh; std::memcpy(&dh, blk, 2);
                const float d = float(dh);
                const int8_t* qs = reinterpret_cast<const int8_t*>(blk+2);
                const float* xv = x + b*32;
                float part = 0.f;
#pragma unroll
                for (int j = 0; j < 32; ++j) part += float(qs[j])*xv[j];
                acc += d*part;
            }
            auto sg = it.get_sub_group();
            acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            if (sg.get_local_id()[0]==0) red[sg.get_group_id()[0]] = acc;
            it.barrier(sycl::access::fence_space::local_space);
            if (tid==0) { float s=0.f; for (int i=0;i<WG/32;++i) s+=red[i]; y[row]=s; }
        });
    });
}

// IQ2_KT fused decode+GEMV over `ne` experts sharing one input vector.
// y is [N * ne]; bases[e] points at expert e's rows.
// Decode expert GEMV, register-tiled over output rows.
//
// At one token the per-lane inner loop reads 8 activations for 8 MACs -- 4 bytes
// per MAC, which puts it on the same ~2.5 TB/s cache ceiling every other kernel
// here hit. Giving a work-group RN rows lets one activation read feed RN MACs;
// the cost is decoding RN rows' trellis instead of one, which is ALU the kernel
// was not using anyway.
template <int WG, int RN>
static sycl::event k_iq2kt_gemv_rt(sycl::queue& q, uint8_t* const* bases,
                                   const float* x, float* y, int64_t K, int64_t N,
                                   int ne, const std::vector<sycl::event>& deps) {
    const int64_t nb = K/256, row_bytes = 4 + nb*68, ng = K/8;
    const int64_t nrt = (N + RN - 1)/RN;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float,1> red(sycl::range<1>(size_t(WG/32)*RN), h);
        h.parallel_for(sycl::nd_range<1>(int64_t(ne)*nrt*WG, WG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int64_t wg = it.get_group(0);
            const int tid = int(it.get_local_id(0));
            const int e = int(wg / nrt);
            const int64_t r0 = (wg % nrt)*RN;
            const uint8_t* eb = bases[e];
            float scale[RN];
            const uint8_t* blk[RN];
#pragma unroll
            for (int r = 0; r < RN; ++r) {
                const int64_t row = sycl::min<int64_t>(r0 + r, N - 1);
                const uint8_t* rp = eb + row*row_bytes;
                std::memcpy(&scale[r], rp, 4);
                blk[r] = rp + 4;
            }
            float acc[RN];
#pragma unroll
            for (int r = 0; r < RN; ++r) acc[r] = 0.f;
            for (int64_t g = tid; g < ng; g += WG) {
                const float* xv = x + 8*g;
                float xr[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) xr[j] = xv[j];
                const int64_t bo = (g >> 5)*68;
                const int ib = int(g & 31);
#pragma unroll
                for (int r = 0; r < RN; ++r) {
                    const uint8_t* b = blk[r] + bo;
                    const uint16_t ql16 = uint16_t(b[4 + 2*ib]) | (uint16_t(b[4 + 2*ib + 1])<<8);
                    const uint32_t seed = uint32_t(ql16) + 4096u;
                    const int sc = (b[(ib/4)%4] >> (4*(ib/16))) & 0xf;
                    const float dl = scale[r] * float(kIq4kValues[sc]) * 1.05f;
                    float part = 0.f;
#pragma unroll
                    for (int j = 0; j < 8; ++j) part += float(trellis_at(seed,j))*xr[j];
                    acc[r] += dl*part;
                }
            }
            auto sg = it.get_sub_group();
            const int sgid = int(sg.get_group_id()[0]);
#pragma unroll
            for (int r = 0; r < RN; ++r) {
                const float v = sycl::reduce_over_group(sg, acc[r], sycl::plus<float>());
                if (sg.get_local_id()[0]==0) red[size_t(sgid)*RN + r] = v;
            }
            it.barrier(sycl::access::fence_space::local_space);
            if (tid == 0)
#pragma unroll
                for (int r = 0; r < RN; ++r) {
                    if (r0 + r >= N) continue;
                    float s2 = 0.f;
                    for (int i = 0; i < WG/32; ++i) s2 += red[size_t(i)*RN + r];
                    y[int64_t(e)*N + r0 + r] = s2;
                }
        });
    });
}

template <int WG = 128>
static sycl::event k_iq2kt_gemv(sycl::queue& q, uint8_t* const* bases,
                                const float* x, float* y, int64_t K, int64_t N,
                                int ne, const std::vector<sycl::event>& deps,
                                const int8_t* ttab) {
    const int64_t nb = K/256, row_bytes = 4 + nb*68, ng = K/8;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float,1> red(sycl::range<1>(WG/32 > 1 ? WG/32 : 1), h);
        h.parallel_for(sycl::nd_range<1>(int64_t(ne)*N*WG, WG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int64_t wg = it.get_group(0);
            const int tid = int(it.get_local_id(0));
            const int e = int(wg / N);
            const int64_t row = wg % N;
            const uint8_t* rp = bases[e] + row*row_bytes;
            float scale; std::memcpy(&scale, rp, 4);
            const uint8_t* blocks = rp + 4;
            float acc = 0.f;
            for (int64_t g = tid; g < ng; g += WG) {
                const uint8_t* b = blocks + (g/32)*68;
                const int ib = int(g % 32);
                const uint16_t ql16 = uint16_t(b[4 + 2*ib]) | (uint16_t(b[4 + 2*ib + 1])<<8);
                const int sc = (b[(ib/4)%4] >> (4*(ib/16))) & 0xf;
                const float dl = scale * float(kIq4kValues[sc]) * 1.05f;
                const float* xv = x + 8*g;
                const uint32_t seed = uint32_t(ql16) + 4096u;
                float part = 0.f;
#pragma unroll
                for (int j = 0; j < 8; ++j) part += float(trellis_at(seed,j))*xv[j];
                acc += dl*part;
            }
            auto sg = it.get_sub_group();
            acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            if constexpr (WG == 32) {
                if (tid==0) y[wg] = acc;
            } else {
                if (sg.get_local_id()[0]==0) red[sg.get_group_id()[0]] = acc;
                it.barrier(sycl::access::fence_space::local_space);
                if (tid==0) { float s=0.f; for (int i=0;i<WG/32;++i) s+=red[i]; y[wg]=s; }
            }
        });
    });
}

// Fused MoE down-projection: out[r] = sum_k w[k] * (W_k . gu_k)[r].
// One work-group per output row, looping the 8 experts inside, so the whole
// down stage is ONE launch instead of 8 GEMVs + 8 axpys. Each expert has its own
// input vector (gu + k*x_stride), which is why the shared-input GEMV cannot be
// reused here.
static sycl::event k_iq2kt_moe_down(sycl::queue& q, const int8_t* ttab, uint8_t* const* bases,
                                    const float* gu, int64_t x_stride,
                                    const float* w, float* out,
                                    int64_t K, int64_t N, int ne) {
    const int64_t nb = K/256, row_bytes = 4 + nb*68, ng = K/8;
    // One work-group per output row, ONE SUB-GROUP PER EXPERT. The experts are
    // independent, so looping them inside a work-group (with two barriers each)
    // serialised 8 independent chunks of work behind 16 barriers. Mapping expert
    // e to sub-group e runs all 8 concurrently and needs a single barrier.
    constexpr int SG = 32;
    const int WG = SG * ne;                       // 256 for the top-8 case
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float,1> red(sycl::range<1>(size_t(ne)), h);
        h.parallel_for(sycl::nd_range<1>(N*WG, WG), [=](sycl::nd_item<1> it)
                       [[sycl::reqd_sub_group_size(SG)]] {
            const int64_t row = it.get_group(0);
            auto sg = it.get_sub_group();
            const int e    = int(sg.get_group_id()[0]);      // expert
            const int lane = int(sg.get_local_id()[0]);
            const uint8_t* rp = bases[e] + row*row_bytes;
            float scale; std::memcpy(&scale, rp, 4);
            const uint8_t* blocks = rp + 4;
            const float* xs = gu + int64_t(e)*x_stride;
            float acc = 0.f;
            for (int64_t g = lane; g < ng; g += SG) {
                const uint8_t* b = blocks + (g/32)*68;
                const int ib = int(g % 32);
                const uint16_t ql16 = uint16_t(b[4+2*ib]) | (uint16_t(b[4+2*ib+1])<<8);
                const uint32_t seed = uint32_t(ql16) + 4096u;
                const int sc = (b[(ib/4)%4] >> (4*(ib/16))) & 0xf;
                const float dl = scale * float(kIq4kValues[sc]) * 1.05f;
                const float* xv = xs + 8*g;
                float part = 0.f;
#pragma unroll
                for (int j = 0; j < 8; ++j) part += float(trellis_at(uint32_t(ql16)+4096u, j))*xv[j];
                acc += dl*part;
            }
            acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            if (lane == 0) red[e] = w[e]*acc;
            it.barrier(sycl::access::fence_space::local_space);
            if (it.get_local_id(0) == 0) {
                float s = 0.f;
                for (int i = 0; i < ne; ++i) s += red[i];
                out[row] = s;
            }
        });
    });
}

// Per-expert variant of the fused down-projection: identical per-lane arithmetic,
// but each expert's weighted row lands in its own output row
// (out[slot[e]*N+row] = w[e]*acc) instead of being summed across experts in SLM.
// k_moe_gather_rows then adds the rows in ascending slot order — the exact order
// the SLM loop used — so the hit/miss split changes WHERE partial sums live, not
// one float operation or its order.
static sycl::event k_iq2kt_moe_down_pe(sycl::queue& q, const int8_t* ttab, uint8_t* const* bases,
                                       const float* gu, int64_t x_stride,
                                       const float* w, const int32_t* slot, float* out,
                                       int64_t K, int64_t N, int ne) {
    (void)ttab;
    const int64_t nb = K/256, row_bytes = 4 + nb*68, ng = K/8;
    constexpr int SG = 32;
    const int WG = SG * ne;
    return q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(N*WG, WG), [=](sycl::nd_item<1> it)
                       [[sycl::reqd_sub_group_size(SG)]] {
            const int64_t row = it.get_group(0);
            auto sg = it.get_sub_group();
            const int e    = int(sg.get_group_id()[0]);
            const int lane = int(sg.get_local_id()[0]);
            const uint8_t* rp = bases[e] + row*row_bytes;
            float scale; std::memcpy(&scale, rp, 4);
            const uint8_t* blocks = rp + 4;
            const float* xs = gu + int64_t(e)*x_stride;
            float acc = 0.f;
            for (int64_t g = lane; g < ng; g += SG) {
                const uint8_t* b = blocks + (g/32)*68;
                const int ib = int(g % 32);
                const uint16_t ql16 = uint16_t(b[4+2*ib]) | (uint16_t(b[4+2*ib+1])<<8);
                const int sc = (b[(ib/4)%4] >> (4*(ib/16))) & 0xf;
                const float dl = scale * float(kIq4kValues[sc]) * 1.05f;
                const float* xv = xs + 8*g;
                float part = 0.f;
#pragma unroll
                for (int j = 0; j < 8; ++j) part += float(trellis_at(uint32_t(ql16)+4096u, j))*xv[j];
                acc += dl*part;
            }
            acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            if (lane == 0) out[int64_t(slot[e])*N + row] = w[e]*acc;
        });
    });
}

// out[row] = sum over slots 0..ne-1 of dn[slot*N+row], ascending slot — the same
// order k_iq2kt_moe_down's final SLM loop uses.
static sycl::event k_moe_gather_rows(sycl::queue& q, const float* dn, float* out,
                                     int64_t N, int ne) {
    return q.parallel_for(sycl::range<1>(size_t(N)), [=](sycl::id<1> i) {
        const int64_t row = int64_t(i[0]);
        float s = 0.f;
        for (int k = 0; k < ne; ++k) s += dn[int64_t(k)*N + row];
        out[row] = s;
    });
}

namespace mat = sycl::ext::oneapi::experimental::matrix;

// IQ2_KT expert GEMM on XMX/DPAS.
//
// The scalar kernel reaches ~7% of this hardware's arithmetic peak, which is what
// caps prompt processing. Structure copied from src/ops/gemm_q4k_xmx.cpp, which
// already does this for Q4_K: dequantise a K-tile of weights into SLM as fp16,
// then feed joint_matrix. IQ2_KT lands on it exactly — its 256-weight block IS
// the BK=256 tile, so one block decode fills one K iteration.
//
//   y[m][row] = sum_k W[row][k] * x[tok[m]][k]
//   WG tile: BN=64 output rows x up to 16 tokens; BK=256 = one IQ2_KT block.
static sycl::event k_iq2kt_gemm_xmx(sycl::queue& q, const uint8_t* base,
                                    const float* x, int64_t x_stride,
                                    const int32_t* tok, int nt,
                                    float* out, int64_t K, int64_t N,
                                    const std::vector<sycl::event>& deps) {
    // BN halved to 32 (B_smem 16 KiB instead of 32) for occupancy, and the
    // work-group widened to 8 sub-groups covering (N-tile x M-group) so the
    // block decode is spread over 128 lanes instead of 64 — the two things the
    // first XMX attempt got wrong.
    constexpr int SG_SIZE = 16, XBN = 32, XBK = 256;
    constexpr int TM = 8, TN = 16, TK = 16;
    constexpr int MG = 2, MTILE = TM*MG;           // up to 16 tokens per WG
    constexpr int NT_WG = XBN / TN;                // 2 N-tiles
    constexpr int WG_ITEMS = NT_WG * MG * SG_SIZE; // 2 x 2 x 16 = 64 lanes
    const int64_t nb = K/256, row_bytes = 4 + nb*68;
    const int64_t wgs_n = N / XBN;
    const int m_tiles = (nt + MTILE - 1) / MTILE;

    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 2> A_smem({MTILE, XBK}, h);
        sycl::local_accessor<sycl::half, 2> B_smem({XBK, XBN}, h);
        sycl::local_accessor<float, 1> C_scratch(size_t(MG)*NT_WG*TM*TN, h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(wgs_n)*m_tiles*WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const int lid  = int(it.get_local_id(0));
            const int64_t wgid = it.get_group(0);
            const int64_t wn = wgid % wgs_n;         // which 64-row output tile
            const int mt = int(wgid / wgs_n);        // which 16-token tile
            const int m0 = mt * MTILE;
            const int M  = sycl::min(MTILE, nt - m0);
            const int sgi   = lid / SG_SIZE;         // 0..3
            const int sg_id  = sgi % NT_WG;          // which N-tile
            const int mgroup = sgi / NT_WG;          // which M-group
            auto sg = it.get_sub_group();

            mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator, TM, TN> acc;
            mat::joint_matrix_fill(sg, acc, 0.0f);

            for (int64_t b = 0; b < nb; ++b) {
                // A tile: the tokens routed here, fp16
                for (int i = lid; i < M*XBK; i += WG_ITEMS) {
                    const int m = i / XBK, kk = i % XBK;
                    A_smem[m][kk] = sycl::half(x[int64_t(tok[m0+m])*x_stride + b*XBK + kk]);
                }
                for (int i = M*XBK + lid; i < MTILE*XBK; i += WG_ITEMS)
                    A_smem[i/XBK][i%XBK] = sycl::half(0.f);
                // B tile: decode this IQ2_KT block for XBN output rows, spread
                // across the whole work-group (each lane takes a slice of the 32
                // weight-groups of its column rather than all of them).
                {
                    const int n_local = lid % XBN;              // column
                    const int half    = lid / XBN;              // 0 or 1
                    const int64_t n_global = wn*XBN + n_local;
                    const uint8_t* rp = base + n_global*row_bytes;
                    float scale; std::memcpy(&scale, rp, 4);
                    const uint8_t* blk = rp + 4 + b*68;
                    for (int ib = half*16; ib < half*16 + 16; ++ib) {
                        const uint16_t ql16 = uint16_t(blk[4+2*ib]) |
                                              (uint16_t(blk[4+2*ib+1])<<8);
                        const uint32_t seed = uint32_t(ql16) + 4096u;
                        const int sc = (blk[(ib/4)%4] >> (4*(ib/16))) & 0xf;
                        const float dl = scale * float(kIq4kValues[sc]) * 1.05f;
#pragma unroll
                        for (int j = 0; j < 8; ++j)
                            B_smem[ib*8 + j][n_local] =
                                sycl::half(dl * float(trellis_at(seed, j)));
                    }
                }
                sycl::group_barrier(it.get_group());
#pragma unroll
                for (int kt = 0; kt < XBK/TK; ++kt) {
                    mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::b,
                                      TK, TN, mat::layout::row_major> b_tile;
                    mat::joint_matrix_load(sg, b_tile,
                        B_smem.get_multi_ptr<sycl::access::decorated::no>() +
                        kt*TK*XBN + sg_id*TN, XBN);
                    mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::a,
                                      TM, TK, mat::layout::row_major> a_tile;
                    mat::joint_matrix_load(sg, a_tile,
                        A_smem.get_multi_ptr<sycl::access::decorated::no>() +
                        int64_t(mgroup)*TM*XBK + kt*TK, XBK);
                    mat::joint_matrix_mad(sg, acc, a_tile, b_tile, acc);
                }
                sycl::group_barrier(it.get_group());
            }
            mat::joint_matrix_store(sg, acc,
                C_scratch.get_multi_ptr<sycl::access::decorated::no>() +
                    int64_t(sgi)*TM*TN, TN, mat::layout::row_major);
            sycl::group_barrier(it.get_group());
            for (int i = lid; i < MG*NT_WG*TM*TN; i += WG_ITEMS) {
                const int s2  = i / (TM*TN);         // which (N-tile, M-group) SG
                const int rr  = i % (TM*TN);
                const int g   = s2 / NT_WG;
                const int nt2 = s2 % NT_WG;
                const int m   = g*TM + rr / TN;
                const int n   = nt2*TN + rr % TN;
                if (m < M)
                    out[int64_t(m0+m)*N + wn*XBN + n] =
                        C_scratch[int64_t(s2)*TM*TN + rr];
            }
        });
    });
}

// ALL of a block's experts in ONE launch.
//
// The per-expert version needs 4 launches per expert: 64+ per block for a
// 2-token speculative verify, 1024 for a 512-token prefill, against ~6 for a
// single-token decode step. That launch overhead was making batched verify 2.5x
// more expensive than plain decode and wiping out the speculative gain.
// Work-group wg maps to (expert e = wg/N, row = wg%N), so one dispatch covers
// every (expert, row) pair.
static sycl::event k_iq2kt_gemm_many(sycl::queue& q, uint8_t* const* bases,
                                     const int32_t* eoff, const int32_t* ecnt,
                                     const int32_t* goff,
                                     const float* x, int64_t x_stride,
                                     const int32_t* tok, float* out,
                                     int64_t K, int64_t N, int ne, int64_t base_off,
                                     const std::vector<sycl::event>& deps) {
    const int64_t nb = K/256, row_bytes = 4 + nb*68, ng = K/8;
    constexpr int WG = 64;
    constexpr int TT = 8;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float,1> red(sycl::range<1>(size_t(WG/32)*TT), h);
        h.parallel_for(sycl::nd_range<1>(int64_t(ne)*N*WG, WG), [=](sycl::nd_item<1> it)
                       [[sycl::reqd_sub_group_size(32)]] {
            const int64_t wg = it.get_group(0);
            const int e = int(wg / N);
            const int64_t row = wg % N;
            const int nt = ecnt[e];
            if (nt <= 0) return;
            const int tid = int(it.get_local_id(0));
            const uint8_t* rp = bases[e] + base_off + row*row_bytes;
            float scale; std::memcpy(&scale, rp, 4);
            const uint8_t* blocks = rp + 4;
            const int32_t* tk = tok + eoff[e];
            float* o = out + int64_t(goff[e])*N;
            for (int t0 = 0; t0 < nt; t0 += TT) {
                const int tn = sycl::min(TT, nt - t0);
                float acc[TT];
                for (int i = 0; i < TT; ++i) acc[i] = 0.f;
                for (int64_t g = tid; g < ng; g += WG) {
                    const uint8_t* b = blocks + (g/32)*68;
                    const int ib = int(g % 32);
                    const uint16_t ql16 = uint16_t(b[4+2*ib]) | (uint16_t(b[4+2*ib+1])<<8);
                    const uint32_t seed = uint32_t(ql16) + 4096u;
                    const int sc = (b[(ib/4)%4] >> (4*(ib/16))) & 0xf;
                    const float dl = scale * float(kIq4kValues[sc]) * 1.05f;
                    float wv[8];
#pragma unroll
                    for (int j = 0; j < 8; ++j) wv[j] = dl * float(trellis_at(seed, j));
                    for (int i = 0; i < tn; ++i) {
                        const float* xv = x + int64_t(tk[t0+i])*x_stride + 8*g;
                        float p = 0.f;
#pragma unroll
                        for (int j = 0; j < 8; ++j) p += wv[j]*xv[j];
                        acc[i] += p;
                    }
                }
                auto sg = it.get_sub_group();
                const int sgid = int(sg.get_group_id()[0]);
                for (int i = 0; i < tn; ++i) {
                    float v = sycl::reduce_over_group(sg, acc[i], sycl::plus<float>());
                    if (sg.get_local_id()[0]==0) red[size_t(sgid)*TT + i] = v;
                }
                it.barrier(sycl::access::fence_space::local_space);
                if (tid == 0)
                    for (int i = 0; i < tn; ++i) {
                        float sum = 0.f;
                        for (int k2 = 0; k2 < WG/32; ++k2) sum += red[size_t(k2)*TT + i];
                        o[int64_t(t0+i)*N + row] = sum;
                    }
                it.barrier(sycl::access::fence_space::local_space);
            }
        });
    });
}

// Batched IQ2_KT GEMM: one expert's weights against NT token vectors.
//
// Decode is the expensive part of a trellis quant, and in a GEMV it is paid once
// per token. Here a work-group decodes a row ONCE into registers and applies it
// to every token routed to this expert, so the trellis cost amortises NT-fold
// and the kernel turns arithmetic-bound. This is what makes prefill scale.
//
//   xs[i] : activation of the i-th token routed here, at x + tok[i]*x_stride
//   out[i*N + row] accumulated
static sycl::event k_iq2kt_gemm_v1(sycl::queue& q, const uint8_t* base,
                                const float* x, int64_t x_stride,
                                const int32_t* tok, int nt,
                                float* out, int64_t K, int64_t N,
                                const std::vector<sycl::event>& deps) {
    const int64_t nb = K/256, row_bytes = 4 + nb*68, ng = K/8;
    constexpr int WG = 64;
    constexpr int TT = 8;                    // token tile held in registers
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float,1> red(sycl::range<1>(size_t(WG/32)*TT), h);
        h.parallel_for(sycl::nd_range<1>(N*WG, WG), [=](sycl::nd_item<1> it)
                       [[sycl::reqd_sub_group_size(32)]] {
            const int64_t row = it.get_group(0);
            const int tid = int(it.get_local_id(0));
            const uint8_t* rp = base + row*row_bytes;
            float scale; std::memcpy(&scale, rp, 4);
            const uint8_t* blocks = rp + 4;
            for (int t0 = 0; t0 < nt; t0 += TT) {
                const int tn = sycl::min(TT, nt - t0);
                float acc[TT];
                for (int i = 0; i < TT; ++i) acc[i] = 0.f;
                for (int64_t g = tid; g < ng; g += WG) {
                    const uint8_t* b = blocks + (g/32)*68;
                    const int ib = int(g % 32);
                    const uint16_t ql16 = uint16_t(b[4+2*ib]) | (uint16_t(b[4+2*ib+1])<<8);
                    const uint32_t seed = uint32_t(ql16) + 4096u;
                    const int sc = (b[(ib/4)%4] >> (4*(ib/16))) & 0xf;
                    const float dl = scale * float(kIq4kValues[sc]) * 1.05f;
                    float wv[8];
#pragma unroll
                    for (int j = 0; j < 8; ++j) wv[j] = dl * float(trellis_at(seed, j));
                    for (int i = 0; i < tn; ++i) {
                        const float* xv = x + int64_t(tok[t0+i])*x_stride + 8*g;
                        float p = 0.f;
#pragma unroll
                        for (int j = 0; j < 8; ++j) p += wv[j]*xv[j];
                        acc[i] += p;
                    }
                }
                auto sg = it.get_sub_group();
                const int sgid = int(sg.get_group_id()[0]);
                for (int i = 0; i < tn; ++i) {
                    float v = sycl::reduce_over_group(sg, acc[i], sycl::plus<float>());
                    if (sg.get_local_id()[0]==0) red[size_t(sgid)*TT + i] = v;
                }
                it.barrier(sycl::access::fence_space::local_space);
                if (tid == 0)
                    for (int i = 0; i < tn; ++i) {
                        float sum = 0.f;
                        for (int k2 = 0; k2 < WG/32; ++k2) sum += red[size_t(k2)*TT + i];
                        out[int64_t(t0+i)*N + row] = sum;
                    }
                it.barrier(sycl::access::fence_space::local_space);
            }
        });
    });
}

// ---- SLM-blocked GEMMs -------------------------------------------------
//
// The _v1 kernels above give each output row its own work-group, so every row
// re-reads the whole activation tile: N * T * K * 4 bytes of cache traffic for
// 2 * N * T * K flops, i.e. 2 bytes per flop. At T=512 that is ~47 TB per
// prefill chunk against 23 TFLOP of useful work, and it — not the ALU — is what
// held prefill at 7% of peak.
//
// The _v2 kernels put RN output rows in one work-group (one row per sub-group)
// and stage the activation tile for a K-chunk in SLM, so the global read is
// amortised over RN rows. Arithmetic and numerics are unchanged: the same
// weights are decoded the same way and summed in the same order per row.

// ---- IE_GLM52_PROBE: device-event overlap probe (log §23 follow-up) --------
// Creates every queue with profiling enabled, records device start/end stamps
// of expert copies (kind 0), descriptor copies (1) and MoE kernels (2) for
// blocks [30,33) — prefill chunk 0 and decode calls 3..5 — and times every host
// wait site. Answers with device timestamps, not phase timers, whether expert
// DMA actually overlaps compute in situ.
static bool g_probe = false;
struct ProbeRec { int card; int block; int kind; uint64_t bytes; sycl::event ev; };
static std::vector<ProbeRec> g_probe_ev;
static double g_w_top=0, g_w_route=0, g_w_mev=0, g_w_drain=0, g_w_fold=0;
static uint64_t g_n_drain=0, g_n_mev=0;
static int g_probe_chunks=0, g_probe_calls=0;
static inline bool probe_blk(int il) { return g_probe && il >= 30 && il < 33; }
static sycl::property_list qprops() {
    return g_probe
        ? sycl::property_list{sycl::property::queue::in_order{},
                              sycl::property::queue::enable_profiling{}}
        : sycl::property_list{sycl::property::queue::in_order{}};
}

static void probe_dump(const char* tag) {
    if (g_probe_ev.empty()) { std::printf("[probe %s] no events captured\n", tag); return; }
    using P = std::pair<uint64_t,uint64_t>;
    auto unify = [](std::vector<P>& v)->uint64_t {
        std::sort(v.begin(), v.end());
        uint64_t busy = 0, cs = 0, ce = 0; bool open = false;
        std::vector<P> merged;
        for (auto& pr : v) {
            if (!open) { cs = pr.first; ce = pr.second; open = true; }
            else if (pr.first <= ce) ce = std::max(ce, pr.second);
            else { merged.push_back({cs, ce}); busy += ce - cs; cs = pr.first; ce = pr.second; }
        }
        if (open) { merged.push_back({cs, ce}); busy += ce - cs; }
        v.swap(merged);
        return busy;
    };
    auto isect = [](const std::vector<P>& a, const std::vector<P>& b)->uint64_t {
        uint64_t s = 0; size_t i = 0, j = 0;
        while (i < a.size() && j < b.size()) {
            const uint64_t lo = std::max(a[i].first, b[j].first);
            const uint64_t hi = std::min(a[i].second, b[j].second);
            if (hi > lo) s += hi - lo;
            if (a[i].second < b[j].second) ++i; else ++j;
        }
        return s;
    };
    for (auto& r : g_probe_ev) r.ev.wait();
    for (int card = 0; card < 2; ++card) {
        for (int blk = 0; blk < 200; ++blk) {
            std::vector<P> cp, kn, ds;
            struct Raw { uint64_t s, e; int k; };
            std::vector<Raw> raw;
            uint64_t t0 = UINT64_MAX, t1 = 0, cpb = 0, subwait = 0; int ncp = 0, nkn = 0;
            for (auto& r : g_probe_ev) {
                if (r.card != card || r.block != blk) continue;
                const uint64_t su = r.ev.get_profiling_info<sycl::info::event_profiling::command_submit>();
                const uint64_t s  = r.ev.get_profiling_info<sycl::info::event_profiling::command_start>();
                const uint64_t e  = r.ev.get_profiling_info<sycl::info::event_profiling::command_end>();
                t0 = std::min(t0, s); t1 = std::max(t1, e);
                raw.push_back({s, e, r.kind});
                if (r.kind == 0) { cp.push_back({s,e}); cpb += r.bytes; ++ncp;
                                   subwait += s - std::min(su, s); }
                else if (r.kind == 1) ds.push_back({s,e});
                else { kn.push_back({s,e}); ++nkn; }
            }
            if (raw.empty()) continue;
            const double cb = double(unify(cp))/1e6, kb = double(unify(kn))/1e6,
                         db = double(unify(ds))/1e6;
            const double ov = double(isect(cp, kn))/1e6;
            std::printf("[probe %s] gpu%d blk%d: span %.2f ms | copies n=%d %.1f MB "
                        "busy %.2f ms (submit->start avg %.0f us) | kernels n=%d busy %.2f ms"
                        " | desc %.3f ms | copy/kernel OVERLAP %.2f ms\n",
                        tag, card, blk, double(t1 - t0)/1e6, ncp, double(cpb)/1e6, cb,
                        ncp ? double(subwait)/1e3/double(ncp) : 0.0, nkn, kb, db, ov);
            if (blk == 31) {
                std::sort(raw.begin(), raw.end(), [](const Raw& a, const Raw& b){ return a.s < b.s; });
                for (auto& r : raw)
                    std::printf("    %c %9.2f -> %9.2f us\n", r.k==0?'C':(r.k==1?'D':'K'),
                                double(r.s - t0)/1e3, double(r.e - t0)/1e3);
            }
        }
    }
    std::printf("[probe %s] host waits (s): top-of-block %.3f | route-d2h %.3f | "
                "desc-slot %.3f (n=%llu) | drain %.3f (n=%llu) | fold %.3f\n",
                tag, g_w_top, g_w_route, g_w_mev, (unsigned long long)g_n_mev,
                g_w_drain, (unsigned long long)g_n_drain, g_w_fold);
    g_probe_ev.clear();
    g_w_top = g_w_route = g_w_mev = g_w_drain = g_w_fold = 0;
    g_n_drain = g_n_mev = 0;
}

static int g_gemm_v2 = 1;   // IE_GLM52_GEMM=0 falls back to the _v1 kernels
static int g_hits_first = 1; // IE_GLM52_HITS_FIRST=0 restores ascending-e order + ramp caps
static int g_dec2 = 0;       // decode: merged syncs + early shared (levels 2 and 3)
static int g_dsplit = 0;     // decode: hit/miss-split GEMV launches (level 2 only)
// Default 3: merged router/xb sync + early shared expert. Every measurement in
// §24-25 was taken at this level and it is the shipping path; 0 restores the
// original two-sync structure.
static int g_dec_lvl = 3;    // IE_GLM52_DEC: 0 = v1, 2 = sync+split, 3 = sync only,
                             // 4 = sync + expert copies ON the compute queue (no
                             // cross-queue deps anywhere — every dep-carrying
                             // submission was measured to block ~1 ms/block in
                             // the UR adapter, and decode copies never
                             // overlapped compute anyway)
                             // 5 = copies stay on the TRANSFER queues, but the
                             // kernels take no dependency on them: the host
                             // waits the transfer queues once per block instead.
                             // Measured (run11): level 4 moved the ~1 ms/block
                             // tax from the dep-carrying kernel submit to the
                             // copy submits themselves (resolve 65 -> 275
                             // ms/token) — the cost is the copy/compute stream
                             // BOUNDARY, wherever it is crossed. Decode overlaps
                             // copy with compute exactly 0.00 ms (probe), so a
                             // host wait costs nothing that was not already
                             // being paid, and it leaves every submission
                             // dependency-free.
// Percent of each VRAM cache reserved for trace-ranked STATIC placement; the
// rest is demand LRU. 87 (7/8) was the campaign default and is right when the
// workload matches the calibration trace — and wrong when it does not: a
// long-prompt decode measured 12.4% hit on gpu0 because only 186 of its 1,485
// slots could adapt. IE_GLM52_STATIC_PCT / the runs sweep set it.
static int g_static_pct = 87;
// Prefill and decode want OPPOSITE cache policies and they run sequentially, so
// the split need not be one number. Prefill touches all 256 experts of every
// block, so an LRU thrashes and fixed static residency wins. Decode reuses a
// narrow set (27.9% of a block's top-8 survives to the next token, and the
// 36-token working set is 74.8 GiB against a 37.4 GiB cache), so it wants the
// largest adaptive pool it can get. Releasing static slots to demand AFTER
// prefill keeps the preloaded experts resident and correct — only their age is
// reset, which makes them LRU candidates rather than pinned.
static int g_static_pct_dec = -1;   // -1 = same as prefill; IE_GLM52_STATIC_DEC
static int g_rp = 1;         // IE_GLM52_RP=0 restores the serial router phase
                             // (xb-bounce wait + top-of-block drain)
// IE_GLM52_STREAM=1: a decode miss is not copied into VRAM at all — the expert
// GEMV is handed the HOST pointer and streams the weights over PCIe as it
// computes. Measured in SELFTEST: the real IQ2_KT GEMV reads host USM at
// 20.5 GB/s (against 26.6 for a DMA copy of the same bytes) and is BYTE-EXACT,
// and host USM allocated in one context is readable by the other card's kernels
// at 24.3 GB/s, so one tier serves both GPUs. This deletes the copy submission,
// the dependency, the host wait and the copy/compute stream boundary — which
// run11 identified as the ~1 ms/block tax — and replaces ~129 ms/token of
// launch+resolve with ~46 ms/token of streaming.
static int g_stream = 0;
// One memcpy per miss instead of two. The gate/up + down split exists so the
// gate/up GEMV can start on 2/3 of the bytes while `down` is still moving. That
// trade is only worth it if the transfer is the constraint — and decode uses
// just 22-28% of the link (run16), so what actually costs is each crossing of
// the copy/compute stream boundary (~1 ms, run11). Fewer, larger copies = fewer
// crossings.
static int g_onecopy = 0;
// Lean event bookkeeping. run17 accounts decode's 184 ms/token completely and
// puts ~42 ms in the residency loop that is NOT copy submits (2 ms) and NOT the
// victim search (0.15 ms). What is left is refcounted sycl::event traffic:
// ~1,600 event copies per token into deps[], c.fill[] and ew.fev[]. Two kinds
// are provably unnecessary — ew.fev/fev_dn are read only by the retired hit/miss
// split path, and a HIT needs no dependency at all because its fill was issued
// in an earlier block and the in-order compute queue already sequenced it.
// Unlike DEC=5 this keeps the miss dependency, so the wait does not move to the
// host.
static int g_leanev = 0;
// Decode's expert descriptors (base pointers + weights) live in malloc_SHARED
// USM: the host writes them every block and the device dereferences them, so
// each block pays page migration in both directions. That shows up as ~41 ms of
// unexplained host time where the writes happen (resolve) and ~121 us per kernel
// submission where the reads happen (launch) — against 4-9 us for the same
// submission in isolation. Prefill already avoids this by staging descriptors in
// host memory and issuing ONE explicit copy per group; this does the same for
// decode. IE_GLM52_DESCSTAGE=0 restores the shared-USM path.
static int g_descstage = 1;
// Eviction is a LINEAR scan for the oldest slot across the whole demand region.
// That was invisible while ~120 ms/token of descriptor migration dominated; with
// that gone (§25) it is the next wall: lowering static residency to raise the
// hit rate grows the scan, and resolve goes 1.1 -> 57-125 ms/token, costing more
// than the extra hits are worth. CLOCK (a rotating hand with a second chance on
// the age bit) is O(1) per miss and approximates LRU closely enough here.
static int g_clock = 0;
// Split the shared expert across both cards. Measured 91.41 vs 91.88 BEFORE
// hits-first (§21b): the extra cross-card transfer cost more than the halved
// work, because the shared expert was already hidden inside the exposed expert
// DMA. Hits-first changed what is exposed, so it is worth re-measuring.
static int g_shtp = 0;
static int g_spin = 1;       // IE_GLM52_SPIN=0 restores blocking event waits
// IE_GLM52_HOSTPROF=1: pure host-side segment stamps for the decode block loop
// (no queue drains, unlike IE_GLM52_PROFILE) — locates host time, not GPU time.
static int g_hostprof = 0;
static double hp_attn=0, hp_route=0, hp_resolve=0, hp_bounce=0, hp_shared=0,
              hp_launch=0, hp_pf=0, hp_fold=0;
static double hp_r_cpy=0, hp_l_dep1=0, hp_l_rest=0;   // sub-segments
static uint64_t hp_tokens=0, hp_bytes=0;
// Settles the one thing the timers cannot: how much of the per-token host time
// is REAL PCIe transfer. Bytes actually fetched / (the segments that wait on
// them) gives an implied rate — at ~53 GB/s aggregate the time is physics, well
// below it the time is overhead.
// Host-side event wait without the Level-Zero interrupt path. ev.wait() parks
// the thread and eats 100-500 us of wake latency per call; decode makes ~3 such
// calls per block x 78 blocks, which is tens of ms per token. zeEventQueryStatus
// (get_info) is a ~1-2 us non-blocking call, so polling turns the wait into
// actual-completion + ~2 us.
static inline void spin_wait(sycl::event ev) {
    if (!g_spin) { ev.wait(); return; }
    while (ev.get_info<sycl::info::event::command_execution_status>()
           != sycl::info::event_command_status::complete)
        __builtin_ia32_pause();
}
static int g_moe_group = 128; // experts per MoE group; IE_GLM52_MOE_GROUP_SWEEP sweeps it

// Register-blocked Q6_0 GEMM.
//
// The row-per-work-group kernel issues ONE activation load per FMA, which is
// what pins it at ~1.3 TFLOP/s against a measured 64 TFLOP/s fp32 ceiling. Here
// a work-group owns a BM x BN tile of C and each lane keeps a TM x TN register
// accumulator, so a K-step costs TM+TN SLM reads for TM*TN FMAs -- 0.25 reads
// per FMA at 8x8 instead of 1.0. Weights are decoded once per tile into SLM.
template <int BM, int BN, int BK, int TM, int TN, bool BIGGRF = false, bool HALF = false>
static sycl::event k_q6_gemm_v3(sycl::queue& q, const uint8_t* w, const float* x,
                                float* y, int64_t K, int64_t N, int64_t row_bytes,
                                int T, const std::vector<sycl::event>& deps) {
    // BK below the 32-weight Q6_0 block makes `BN*(BK/32)` zero, so the weight
    // staging loop never runs and the kernel computes fast garbage. It measured
    // 10365 GF/s that way against the real 9107 -- a trap worth closing.
    static_assert(BK >= 32 && BK % 32 == 0, "BK must be a multiple of the Q6_0 block");
    static_assert(TM*TN <= 64, "above 64 accumulators the register file spills: "
                               "8x16 measured 436 GF/s against 8x8's 9107");
    constexpr int MT = BM/TM, NT = BN/TN, WG = MT*NT;
    constexpr int PA = BM + 1, PB = BN + 1;      // odd strides: no SLM bank conflicts
    const int64_t nb = K/32;
    const int64_t ntm = (T + BM - 1)/BM, ntn = (N + BN - 1)/BN;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        // SLM tiles in fp16 halve the traffic the register tiling is bounded by.
        // Accumulation stays fp32; the rounding is ~5e-4 relative against a weight
        // that Q6_0 has already quantised to ~1.6e-2, so it adds ~3% to an error
        // that is already there.
        using Sty = std::conditional_t<HALF, sycl::half, float>;
        sycl::local_accessor<Sty,1> As(sycl::range<1>(size_t(BK)*PA), h);
        sycl::local_accessor<Sty,1> Bs(sycl::range<1>(size_t(BK)*PB), h);
        auto body = [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int64_t gid = it.get_group(0);
            const int64_t m0 = (gid / ntn)*BM, n0 = (gid % ntn)*BN;
            const int lid = int(it.get_local_id(0));
            const int lm = lid % MT, ln = lid / MT;
            float acc[TM][TN];
#pragma unroll
            for (int i=0;i<TM;++i)
#pragma unroll
                for (int j=0;j<TN;++j) acc[i][j] = 0.f;
            for (int64_t k0 = 0; k0 < K; k0 += BK) {
                it.barrier(sycl::access::fence_space::local_space);
                for (int idx = lid; idx < BM*BK; idx += WG) {
                    const int mm = idx / BK, kk = idx - mm*BK;   // lanes stride k: coalesced
                    const int64_t row = m0 + mm;
                    As[size_t(kk)*PA + mm] = Sty((row < T) ? x[row*K + k0 + kk] : 0.f);
                }
                for (int idx = lid; idx < BN*(BK/32); idx += WG) {
                    const int kb = idx / BN, nn = idx - kb*BN;   // lanes stride n: contiguous SLM
                    const int64_t row = n0 + nn, b = k0/32 + kb;
                    if (row >= N) {
#pragma unroll
                        for (int j = 0; j < 32; ++j) Bs[size_t(kb*32+j)*PB + nn] = Sty(0.f);
                        continue;
                    }
                    const uint8_t* rp = w + row*row_bytes;
                    const sycl::half* pd = reinterpret_cast<const sycl::half*>(rp);
                    const uint8_t* pqs = rp + nb*2;
                    const uint8_t* pqh = pqs + K/2;
                    const float d = float(pd[b]);
                    uint32_t qs[4], qh[2];
                    std::memcpy(qs, pqs + b*16, 16);
                    std::memcpy(qh, pqh + b*8, 8);
#pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        const uint32_t sb = (qs[j>>2] >> (8*(j&3))) & 0xFFu;
                        const uint32_t hb = (qh[(j>>2)&1] >> (8*(j&3))) & 0xFFu;
                        const int l0 = int(sb & 0x0F), l1 = int(sb >> 4);
                        const int h0 = int((hb >> (4*(j/8))) & 3);
                        const int h1 = int((hb >> (4*(j/8)+2)) & 3);
                        Bs[size_t(kb*32+j)*PB + nn]    = Sty(d * float((l0 | (h0<<4)) - 32));
                        Bs[size_t(kb*32+16+j)*PB + nn] = Sty(d * float((l1 | (h1<<4)) - 32));
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
#pragma unroll 4
                for (int kk = 0; kk < BK; ++kk) {
                    float a[TM], b[TN];
#pragma unroll
                    for (int i=0;i<TM;++i) a[i] = float(As[size_t(kk)*PA + lm*TM + i]);
#pragma unroll
                    for (int j=0;j<TN;++j) b[j] = float(Bs[size_t(kk)*PB + ln*TN + j]);
#pragma unroll
                    for (int i=0;i<TM;++i)
#pragma unroll
                        for (int j=0;j<TN;++j) acc[i][j] += a[i]*b[j];
                }
            }
#pragma unroll
            for (int i=0;i<TM;++i) {
                const int64_t row = m0 + lm*TM + i;
                if (row >= T) continue;
#pragma unroll
                for (int j=0;j<TN;++j) {
                    const int64_t col = n0 + ln*TN + j;
                    if (col < N) y[row*N + col] = acc[i][j];
                }
            }
        };
        // 4x4 is the largest tile that fits the default 128-GRF budget; 8x8
        // spilled hard. The doubled budget is the only way to test whether a
        // bigger register tile actually pays.
        const sycl::nd_range<1> nr(ntm*ntn*WG, WG);
        if constexpr (BIGGRF)
            h.parallel_for(nr, sycl::ext::oneapi::experimental::properties{
                               sycl::ext::intel::experimental::grf_size<256>}, body);
        else h.parallel_for(nr, body);
    });
}

// Q6_0 dense GEMM on XMX/DPAS.
//
// XMX cannot help the IQ2_KT expert GEMM: its trellis costs ~8 scalar ops per
// weight, so once the MACs run ~8x faster the decode becomes the critical path
// (measured 2946-4321 GF/s against the scalar 6048). Q6_0 decode is ~2 ops per
// weight, an order of magnitude less relative to the MACs it feeds, which is
// exactly the case where a matrix engine should pay.
//
// A and B are fp16, accumulation fp32.
template <int BM, int BN, int BK>
static sycl::event k_q6_gemm_xmx(sycl::queue& q, const uint8_t* w, const float* x,
                                 float* y, int64_t K, int64_t N, int64_t row_bytes,
                                 int T, const std::vector<sycl::event>& deps) {
    namespace mat = sycl::ext::oneapi::experimental::matrix;
    constexpr int SG = 16, TM = 8, TN = 16, TK = 16;
    constexpr int NT = BN/TN, MG = BM/TM, WG = NT*SG, KT = BK/TK;
    static_assert(BK % 32 == 0, "BK must be a multiple of the Q6_0 block");
    const int64_t nb = K/32;
    const int64_t ntm = (T + BM - 1)/BM, ntn = (N + BN - 1)/BN;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half,1> As(sycl::range<1>(size_t(BM)*BK), h);
        sycl::local_accessor<sycl::half,1> Bs(sycl::range<1>(size_t(BK)*BN), h);
        sycl::local_accessor<float,1>      Cs(sycl::range<1>(size_t(NT)*TM*TN), h);
        h.parallel_for(sycl::nd_range<1>(ntm*ntn*WG, WG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const int64_t gid = it.get_group(0);
            const int64_t m0 = (gid / ntn)*BM, n0 = (gid % ntn)*BN;
            const int lid = int(it.get_local_id(0));
            const int sg_id = lid / SG, lane = lid % SG;
            auto sg = it.get_sub_group();

            mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator, TM, TN> acc[MG];
#pragma unroll
            for (int g = 0; g < MG; ++g) mat::joint_matrix_fill(sg, acc[g], 0.0f);

            for (int64_t k0 = 0; k0 < K; k0 += BK) {
                it.barrier(sycl::access::fence_space::local_space);
                for (int idx = lid; idx < BM*BK; idx += WG) {
                    const int mm = idx / BK, kk = idx - mm*BK;
                    const int64_t row = m0 + mm;
                    As[size_t(mm)*BK + kk] =
                        sycl::half((row < T) ? x[row*K + k0 + kk] : 0.f);
                }
                for (int idx = lid; idx < BN*(BK/32); idx += WG) {
                    const int kb = idx / BN, nn = idx - kb*BN;
                    const int64_t row = n0 + nn, b = k0/32 + kb;
                    if (row >= N) {
#pragma unroll
                        for (int j = 0; j < 32; ++j)
                            Bs[size_t(kb*32+j)*BN + nn] = sycl::half(0.f);
                        continue;
                    }
                    const uint8_t* rp = w + row*row_bytes;
                    const sycl::half* pd = reinterpret_cast<const sycl::half*>(rp);
                    const uint8_t* pqs = rp + nb*2;
                    const uint8_t* pqh = pqs + K/2;
                    const float d = float(pd[b]);
                    uint32_t qs[4], qh[2];
                    std::memcpy(qs, pqs + b*16, 16);
                    std::memcpy(qh, pqh + b*8, 8);
#pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        const uint32_t sb = (qs[j>>2] >> (8*(j&3))) & 0xFFu;
                        const uint32_t hb = (qh[(j>>2)&1] >> (8*(j&3))) & 0xFFu;
                        const int l0 = int(sb & 0x0F), l1 = int(sb >> 4);
                        const int h0 = int((hb >> (4*(j/8))) & 3);
                        const int h1 = int((hb >> (4*(j/8)+2)) & 3);
                        Bs[size_t(kb*32+j)*BN + nn]    = sycl::half(d * float((l0 | (h0<<4)) - 32));
                        Bs[size_t(kb*32+16+j)*BN + nn] = sycl::half(d * float((l1 | (h1<<4)) - 32));
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                for (int kt = 0; kt < KT; ++kt) {
                    mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::b,
                                      TK, TN, mat::layout::row_major> b_tile;
                    mat::joint_matrix_load(sg, b_tile,
                        Bs.get_multi_ptr<sycl::access::decorated::no>() +
                        size_t(kt)*TK*BN + size_t(sg_id)*TN, BN);
#pragma unroll
                    for (int g = 0; g < MG; ++g) {
                        mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::a,
                                          TM, TK, mat::layout::row_major> a_tile;
                        mat::joint_matrix_load(sg, a_tile,
                            As.get_multi_ptr<sycl::access::decorated::no>() +
                            size_t(g)*TM*BK + size_t(kt)*TK, BK);
                        mat::joint_matrix_mad(sg, acc[g], a_tile, b_tile, acc[g]);
                    }
                }
            }
            // Store one M-group at a time through a small SLM scratch so rows
            // past T and columns past N are masked off.
            for (int g = 0; g < MG; ++g) {
                it.barrier(sycl::access::fence_space::local_space);
                mat::joint_matrix_store(sg, acc[g],
                    Cs.get_multi_ptr<sycl::access::decorated::no>() +
                    size_t(sg_id)*TM*TN, TN, mat::layout::row_major);
                it.barrier(sycl::access::fence_space::local_space);
                for (int i = lane; i < TM*TN; i += SG) {
                    const int rr = i / TN, cc = i - rr*TN;
                    const int64_t row = m0 + g*TM + rr;
                    const int64_t col = n0 + sg_id*TN + cc;
                    if (row < T && col < N)
                        y[row*N + col] = Cs[size_t(sg_id)*TM*TN + i];
                }
            }
        });
    });
}

// Register-blocked IQ2_KT expert GEMM -- same tiling as k_q6_gemm_v3, with the
// trellis decoded once per (row, K-tile) into SLM. BM is the token tile, and in
// prefill a single expert only sees about T/32 tokens, so BM wants to be small.
template <int BM, int BN, int BK, int TM, int TN>
static sycl::event k_iq2kt_gemm_v3(sycl::queue& q, const uint8_t* base,
                                   const float* x, int64_t x_stride,
                                   const int32_t* tok, int nt,
                                   float* out, int64_t K, int64_t N,
                                   const std::vector<sycl::event>& deps) {
    constexpr int MT = BM/TM, NT = BN/TN, WG = MT*NT;
    constexpr int PA = BM + 1, PB = BN + 1;
    const int64_t nb = K/256, row_bytes = 4 + nb*68;
    const int64_t ntm = (nt + BM - 1)/BM, ntn = (N + BN - 1)/BN;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float,1> As(sycl::range<1>(size_t(BK)*PA), h);
        sycl::local_accessor<float,1> Bs(sycl::range<1>(size_t(BK)*PB), h);
        h.parallel_for(sycl::nd_range<1>(ntm*ntn*WG, WG), [=](sycl::nd_item<1> it)
                       [[sycl::reqd_sub_group_size(32)]] {
            const int64_t gid = it.get_group(0);
            const int64_t m0 = (gid / ntn)*BM, n0 = (gid % ntn)*BN;
            const int lid = int(it.get_local_id(0));
            const int lm = lid % MT, ln = lid / MT;
            float acc[TM][TN];
#pragma unroll
            for (int i=0;i<TM;++i)
#pragma unroll
                for (int j=0;j<TN;++j) acc[i][j] = 0.f;
            for (int64_t k0 = 0; k0 < K; k0 += BK) {
                it.barrier(sycl::access::fence_space::local_space);
                for (int idx = lid; idx < BM*BK; idx += WG) {
                    const int mm = idx / BK, kk = idx - mm*BK;
                    const int64_t row = m0 + mm;
                    As[size_t(kk)*PA + mm] = (row < nt)
                        ? x[int64_t(tok[row])*x_stride + k0 + kk] : 0.f;
                }
                for (int idx = lid; idx < BN*(BK/8); idx += WG) {
                    const int gg = idx / BN, nn = idx - gg*BN;
                    const int64_t row = n0 + nn;
                    if (row >= N) {
#pragma unroll
                        for (int j = 0; j < 8; ++j) Bs[size_t(gg*8+j)*PB + nn] = 0.f;
                        continue;
                    }
                    const uint8_t* rp = base + row*row_bytes;
                    float scale; std::memcpy(&scale, rp, 4);
                    const int64_t g = (k0 >> 3) + gg;
                    const uint8_t* b = rp + 4 + (g >> 5)*68;
                    const int ib = int(g & 31);
                    const uint16_t ql16 = uint16_t(b[4+2*ib]) | (uint16_t(b[4+2*ib+1])<<8);
                    const uint32_t seed = uint32_t(ql16) + 4096u;
                    const int sc = (b[(ib/4)%4] >> (4*(ib/16))) & 0xf;
                    const float dl = scale * float(kIq4kValues[sc]) * 1.05f;
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        Bs[size_t(gg*8+j)*PB + nn] = dl * float(trellis_at(seed, j));
                }
                it.barrier(sycl::access::fence_space::local_space);
#pragma unroll 4
                for (int kk = 0; kk < BK; ++kk) {
                    float a[TM], b[TN];
#pragma unroll
                    for (int i=0;i<TM;++i) a[i] = As[size_t(kk)*PA + lm*TM + i];
#pragma unroll
                    for (int j=0;j<TN;++j) b[j] = Bs[size_t(kk)*PB + ln*TN + j];
#pragma unroll
                    for (int i=0;i<TM;++i)
#pragma unroll
                        for (int j=0;j<TN;++j) acc[i][j] += a[i]*b[j];
                }
            }
#pragma unroll
            for (int i=0;i<TM;++i) {
                const int64_t row = m0 + lm*TM + i;
                if (row >= nt) continue;
#pragma unroll
                for (int j=0;j<TN;++j) {
                    const int64_t col = n0 + ln*TN + j;
                    if (col < N) out[row*N + col] = acc[i][j];
                }
            }
        });
    });
}

// Shape dispatch. The register-blocked kernel needs a full BM x BN tile to pay
// for itself, so anything narrower than the tile (the 2-4 token MTP verify
// window, or K not divisible by BK) still goes to the row-per-work-group kernel.
// Batched MLA-absorb GEMV. wk_b and wv_b are 64 narrow slices each, and prefill
// was launching one kernel PER TOKEN for them -- 2,000 launches per block, twice,
// or ~300,000 per prefill pass. Adding the token dimension to the grid makes it
// one launch, and a TT-token register tile decodes each weight once for TT rows.
static sycl::event k_q6_gemv_multi_small_T(sycl::queue& q, const uint8_t* w,
        const float* x, int64_t x_slice, int64_t x_tok,
        float* y, int64_t y_slice, int64_t y_tok,
        int64_t K, int64_t N, int64_t row_bytes, int nslice, int T) {
    const int64_t nb = K/32;
    constexpr int WG = 128, TT = 4;
    const int64_t rows = int64_t(nslice)*N;
    const int64_t ntile = (T + TT - 1)/TT;
    const int64_t total = rows*ntile;
    return q.parallel_for(sycl::nd_range<1>(((total + WG - 1)/WG)*WG, WG),
                          [=](sycl::nd_item<1> it) {
        const int64_t gid = it.get_global_id(0);
        if (gid >= total) return;
        const int64_t tile = gid / rows, r = gid - tile*rows;
        const int s = int(r / N);
        const uint8_t* rp = w + r*row_bytes;
        const sycl::half* pd = reinterpret_cast<const sycl::half*>(rp);
        const uint8_t* pqs = rp + nb*2;
        const uint8_t* pqh = pqs + K/2;
        const int t0 = int(tile)*TT;
        const int tn = sycl::min(TT, T - t0);
        float acc[TT];
#pragma unroll
        for (int i = 0; i < TT; ++i) acc[i] = 0.f;
        for (int64_t b = 0; b < nb; ++b) {
            const float d = float(pd[b]);
            uint32_t qs[4], qh[2];
            std::memcpy(qs, pqs + b*16, 16);
            std::memcpy(qh, pqh + b*8, 8);
            float wv[32];
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const uint32_t sb = (qs[j>>2] >> (8*(j&3))) & 0xFFu;
                const uint32_t hb = (qh[(j>>2)&1] >> (8*(j&3))) & 0xFFu;
                const int l0 = int(sb & 0x0F), l1 = int(sb >> 4);
                const int h0 = int((hb >> (4*(j/8))) & 3);
                const int h1 = int((hb >> (4*(j/8)+2)) & 3);
                wv[j]      = float((l0 | (h0<<4)) - 32);
                wv[j + 16] = float((l1 | (h1<<4)) - 32);
            }
            for (int i = 0; i < tn; ++i) {
                const float* xv = x + int64_t(t0+i)*x_tok + int64_t(s)*x_slice + b*32;
                float part = 0.f;
#pragma unroll
                for (int j = 0; j < 32; ++j) part += wv[j]*xv[j];
                acc[i] += d*part;
            }
        }
        const int64_t row = r % N;
        for (int i = 0; i < tn; ++i)
            y[int64_t(t0+i)*y_tok + int64_t(s)*y_slice + row] = acc[i];
    });
}

// Register-blocked Q8_0 router GEMM.
//
// The router was the last hot GEMM still using the row-per-work-group shape that
// every other kernel here has been moved off: ~1 TFLOP/s, 1.29 s of a 22.1 s
// prefill. Same tiling as k_q6_gemm_v3; Q8_0 decode is one multiply per weight.
template <int BM, int BN, int BK, int TM, int TN>
static sycl::event k_q8_gemm_v3(sycl::queue& q, const uint8_t* w, const float* x,
                                float* y, int64_t K, int64_t N, int T) {
    constexpr int MT = BM/TM, NT = BN/TN, WG = MT*NT;
    constexpr int PA = BM + 1, PB = BN + 1;
    static_assert(BK % 32 == 0, "BK must be a multiple of the Q8_0 block");
    const int64_t nb = K/32, row_bytes = nb*34;
    const int64_t ntm = (T + BM - 1)/BM, ntn = (N + BN - 1)/BN;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float,1> As(sycl::range<1>(size_t(BK)*PA), h);
        sycl::local_accessor<float,1> Bs(sycl::range<1>(size_t(BK)*PB), h);
        h.parallel_for(sycl::nd_range<1>(ntm*ntn*WG, WG), [=](sycl::nd_item<1> it)
                       [[sycl::reqd_sub_group_size(32)]] {
            const int64_t gid = it.get_group(0);
            const int64_t m0 = (gid / ntn)*BM, n0 = (gid % ntn)*BN;
            const int lid = int(it.get_local_id(0));
            const int lm = lid % MT, ln = lid / MT;
            float acc[TM][TN];
#pragma unroll
            for (int i=0;i<TM;++i)
#pragma unroll
                for (int j=0;j<TN;++j) acc[i][j] = 0.f;
            for (int64_t k0 = 0; k0 < K; k0 += BK) {
                it.barrier(sycl::access::fence_space::local_space);
                for (int idx = lid; idx < BM*BK; idx += WG) {
                    const int mm = idx / BK, kk = idx - mm*BK;
                    const int64_t row = m0 + mm;
                    As[size_t(kk)*PA + mm] = (row < T) ? x[row*K + k0 + kk] : 0.f;
                }
                for (int idx = lid; idx < BN*(BK/32); idx += WG) {
                    const int kb = idx / BN, nn = idx - kb*BN;
                    const int64_t row = n0 + nn, b = k0/32 + kb;
                    if (row >= N) {
#pragma unroll
                        for (int j = 0; j < 32; ++j) Bs[size_t(kb*32+j)*PB + nn] = 0.f;
                        continue;
                    }
                    const uint8_t* blk = w + row*row_bytes + b*34;
                    sycl::half dh; std::memcpy(&dh, blk, 2);
                    const float d = float(dh);
                    const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
#pragma unroll
                    for (int j = 0; j < 32; ++j)
                        Bs[size_t(kb*32+j)*PB + nn] = d * float(qs[j]);
                }
                it.barrier(sycl::access::fence_space::local_space);
#pragma unroll 4
                for (int kk = 0; kk < BK; ++kk) {
                    float a[TM], b[TN];
#pragma unroll
                    for (int i=0;i<TM;++i) a[i] = As[size_t(kk)*PA + lm*TM + i];
#pragma unroll
                    for (int j=0;j<TN;++j) b[j] = Bs[size_t(kk)*PB + ln*TN + j];
#pragma unroll
                    for (int i=0;i<TM;++i)
#pragma unroll
                        for (int j=0;j<TN;++j) acc[i][j] += a[i]*b[j];
                }
            }
#pragma unroll
            for (int i=0;i<TM;++i) {
                const int64_t row = m0 + lm*TM + i;
                if (row >= T) continue;
#pragma unroll
                for (int j=0;j<TN;++j) {
                    const int64_t col = n0 + ln*TN + j;
                    if (col < N) y[row*N + col] = acc[i][j];
                }
            }
        });
    });
}

// Router top-k on the GPU.
//
// Prefill was doing this on one CPU core: 256 sigmoids, a bias add and a
// partial_sort of 256 per token per block -- 152,000 sorts and 39M exp() calls
// for a 2000-token prompt, 1.5 s of a 28 s prefill. It also forced a 2 MB
// device->host copy of the full logit matrix per block; only T x 8 indices and
// weights are actually needed, which is 16x less.
//
// Selection is by sigmoid(logit) + expert_bias, exactly as the CPU path; the
// returned weights are the UNBIASED sigmoids, normalised over the top-k and
// scaled. Ties break to the lower expert id (the CPU's partial_sort leaves ties
// unspecified, and exact ties between distinct logits do not occur in practice).
static sycl::event k_router_topk(sycl::queue& q, const float* rl, const float* bias,
                                 int32_t* top, float* topw, int NE, int TOPK,
                                 float wscale, int T) {
    constexpr int SG = 32;
    const int per = NE / SG;
    return q.parallel_for(sycl::nd_range<1>(size_t(T)*SG, SG), [=](sycl::nd_item<1> it)
                          [[sycl::reqd_sub_group_size(SG)]] {
        const int t = int(it.get_group(0));
        auto sg = it.get_sub_group();
        const int lane = int(sg.get_local_id()[0]);
        const float* rlt = rl + int64_t(t)*NE;
        float sc[8];
        for (int i = 0; i < per; ++i) {
            const float v = rlt[lane*per + i];
            sc[i] = 1.f/(1.f + sycl::exp(-v)) + bias[lane*per + i];
        }
        float wsum = 0.f;
        for (int k = 0; k < TOPK; ++k) {
            float bv = -INFINITY; int bi = 0;
            for (int i = 0; i < per; ++i) if (sc[i] > bv) { bv = sc[i]; bi = i; }
            const float m = sycl::reduce_over_group(sg, bv, sycl::maximum<float>());
            const int cand = (bv == m) ? (lane*per + bi) : NE;
            const int win  = sycl::reduce_over_group(sg, cand, sycl::minimum<int>());
            if (win / per == lane) sc[win - lane*per] = -INFINITY;
            const float wp = 1.f/(1.f + sycl::exp(-rlt[win]));
            wsum += wp;
            if (lane == 0) { top[int64_t(t)*TOPK + k] = win; topw[int64_t(t)*TOPK + k] = wp; }
        }
        if (lane == 0)
            for (int k = 0; k < TOPK; ++k)
                topw[int64_t(t)*TOPK + k] = topw[int64_t(t)*TOPK + k]/wsum*wscale;
    });
}

// Batched Q8_0 router GEMM. The router was a per-token GEMV loop -- at a
// 2000-token chunk that is 2,000 launches per block, 152,000 per prefill pass,
// for 6.3 GFLOP of actual work.
static sycl::event k_q8_gemm(sycl::queue& q, const uint8_t* w, const float* x,
                             float* y, int64_t K, int64_t N, int T) {
    const int64_t nb = K/32, row_bytes = nb*34;
    constexpr int WG = 64, TT = 8;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float,1> red(sycl::range<1>(size_t(WG/32)*TT), h);
        h.parallel_for(sycl::nd_range<1>(N*WG, WG), [=](sycl::nd_item<1> it)
                       [[sycl::reqd_sub_group_size(32)]] {
            const int64_t row = it.get_group(0);
            const int tid = int(it.get_local_id(0));
            const uint8_t* rp = w + row*row_bytes;
            auto sg = it.get_sub_group();
            const int sgid = int(sg.get_group_id()[0]);
            for (int t0 = 0; t0 < T; t0 += TT) {
                const int tn = sycl::min(TT, T - t0);
                float acc[TT];
                for (int i = 0; i < TT; ++i) acc[i] = 0.f;
                for (int64_t b = tid; b < nb; b += WG) {
                    const uint8_t* blk = rp + b*34;
                    sycl::half dh; std::memcpy(&dh, blk, 2);
                    const float d = float(dh);
                    const int8_t* qs = reinterpret_cast<const int8_t*>(blk+2);
                    for (int i = 0; i < tn; ++i) {
                        const float* xv = x + int64_t(t0+i)*K + b*32;
                        float part = 0.f;
#pragma unroll
                        for (int j = 0; j < 32; ++j) part += float(qs[j])*xv[j];
                        acc[i] += d*part;
                    }
                }
                for (int i = 0; i < tn; ++i) {
                    const float v = sycl::reduce_over_group(sg, acc[i], sycl::plus<float>());
                    if (sg.get_local_id()[0]==0) red[size_t(sgid)*TT + i] = v;
                }
                it.barrier(sycl::access::fence_space::local_space);
                if (tid == 0)
                    for (int i = 0; i < tn; ++i) {
                        float sm = 0.f;
                        for (int k = 0; k < WG/32; ++k) sm += red[size_t(k)*TT + i];
                        y[int64_t(t0+i)*N + row] = sm;
                    }
                it.barrier(sycl::access::fence_space::local_space);
            }
        });
    });
}

template <int BM_, int TM_> struct MoeTile { static constexpr int bm = BM_, tm = TM_; };

// Batched IQ2_KT expert GEMM: ONE launch covers every active expert.
//
// In prefill a single expert only sees about T/32 of the chunk, so a per-expert
// launch leaves the register-blocked tiling with a couple of thousand
// work-items and it loses to the row-per-work-group kernel on occupancy alone.
// Tiling over (expert, token-tile, row-tile) in one grid restores it: 256
// experts x 16 row-tiles is half a million work-items. It also collapses the
// launch count from 5 per expert to 5 per block, which is what made the
// speculative-decode verify cost 2.5x a decode step.
//
// `tokall` selects the source row: gate/up gather by the packed routing list,
// down reads the packed gate*up rows written by this same kernel.
// Batched IQ2_KT expert GEMM on XMX/DPAS.
//
// The first XMX attempt (§12k) used 44 KiB of SLM against 64 lanes and lost to
// the scalar path at every size. Everything since then says why: occupancy is
// set by the SLM footprint, and the winning scalar shapes are wide-N with a
// small K tile. This one keeps A and B tiles small (8 + 32 KiB), uses SG_SIZE=16
// as the Xe DPAS path requires, and stores accumulators straight to global
// instead of through a C scratch.
//
// A and B are fp16, accumulation fp32. That rounds an activation and a decoded
// weight to ~5e-4 relative, against an IQ2_KT weight already quantised to ~1e-1
// -- but it IS a numerics change and is verified against the reference prompt.
template <int BM, int BN, int BK>
static sycl::event k_iq2kt_moe_xmx(sycl::queue& q,
        uint8_t* const* bases, const int32_t* eoff, const int32_t* ecnt,
        const int32_t* goff, const int32_t* tiles, int ntiles, int64_t woff,
        const int32_t* tokall, const float* xg, int64_t x_stride,
        float* og, int64_t K, int64_t N,
        const std::vector<sycl::event>& deps) {
    namespace mat = sycl::ext::oneapi::experimental::matrix;
    constexpr int SG = 16, TM = 8, TN = 16, TK = 16;
    constexpr int NT = BN/TN, MG = BM/TM, WG = NT*SG, KT = BK/TK;
    static_assert(BK % 8 == 0 && BK <= 256, "BK must tile the 8-weight trellis group");
    const int64_t nb = K/256, row_bytes = 4 + nb*68;
    const int64_t ntn = (N + BN - 1)/BN;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half,1> As(sycl::range<1>(size_t(BM)*BK), h);
        sycl::local_accessor<sycl::half,1> Bs(sycl::range<1>(size_t(BK)*BN), h);
        h.parallel_for(sycl::nd_range<1>(size_t(ntiles)*size_t(ntn)*WG, WG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const int64_t gid = it.get_group(0);
            const int ti = int(gid / ntn);
            const int64_t n0 = (gid % ntn)*BN;
            const int ke = tiles[2*ti], m0 = tiles[2*ti + 1];
            const uint8_t* wb = bases[ke] + woff;
            const int nt = ecnt[ke], eo = eoff[ke], go = goff[ke];
            const int lid = int(it.get_local_id(0));
            const int sg_id = lid / SG;
            auto sg = it.get_sub_group();

            mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator, TM, TN> acc[MG];
#pragma unroll
            for (int g = 0; g < MG; ++g) mat::joint_matrix_fill(sg, acc[g], 0.0f);

            for (int64_t k0 = 0; k0 < K; k0 += BK) {
                it.barrier(sycl::access::fence_space::local_space);
                for (int idx = lid; idx < BM*BK; idx += WG) {
                    const int mm = idx / BK, kk = idx - mm*BK;
                    const int m = m0 + mm;
                    float v = 0.f;
                    if (m < nt) {
                        const int64_t srow = tokall ? int64_t(tokall[eo + m])
                                                    : int64_t(go + m);
                        v = xg[srow*x_stride + k0 + kk];
                    }
                    As[size_t(mm)*BK + kk] = sycl::half(v);
                }
                for (int idx = lid; idx < BN*(BK/8); idx += WG) {
                    const int gg = idx / BN, nn = idx - gg*BN;
                    const int64_t row = n0 + nn;
                    if (row >= N) {
#pragma unroll
                        for (int j = 0; j < 8; ++j) Bs[size_t(gg*8+j)*BN + nn] = sycl::half(0.f);
                        continue;
                    }
                    const uint8_t* rp = wb + row*row_bytes;
                    float scale; std::memcpy(&scale, rp, 4);
                    const int64_t gI = (k0 >> 3) + gg;
                    const uint8_t* b = rp + 4 + (gI >> 5)*68;
                    const int ib = int(gI & 31);
                    const uint16_t ql16 = uint16_t(b[4+2*ib]) | (uint16_t(b[4+2*ib+1])<<8);
                    const uint32_t seed = uint32_t(ql16) + 4096u;
                    const int sc = (b[(ib/4)%4] >> (4*(ib/16))) & 0xf;
                    const float dl = scale * float(kIq4kValues[sc]) * 1.05f;
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        Bs[size_t(gg*8+j)*BN + nn] = sycl::half(dl * float(trellis_at(seed, j)));
                }
                it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                for (int kt = 0; kt < KT; ++kt) {
                    mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::b,
                                      TK, TN, mat::layout::row_major> b_tile;
                    mat::joint_matrix_load(sg, b_tile,
                        Bs.get_multi_ptr<sycl::access::decorated::no>() +
                        size_t(kt)*TK*BN + size_t(sg_id)*TN, BN);
#pragma unroll
                    for (int g = 0; g < MG; ++g) {
                        mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::a,
                                          TM, TK, mat::layout::row_major> a_tile;
                        mat::joint_matrix_load(sg, a_tile,
                            As.get_multi_ptr<sycl::access::decorated::no>() +
                            size_t(g)*TM*BK + size_t(kt)*TK, BK);
                        mat::joint_matrix_mad(sg, acc[g], a_tile, b_tile, acc[g]);
                    }
                }
            }
            it.barrier(sycl::access::fence_space::local_space);
            // Straight to global. Rows past `nt` land in the pad the caller
            // leaves at the end of the packed buffer.
#pragma unroll
            for (int g = 0; g < MG; ++g)
                mat::joint_matrix_store(sg, acc[g],
                    sycl::address_space_cast<sycl::access::address_space::global_space,
                                             sycl::access::decorated::no>(
                        og + int64_t(go + m0 + g*TM)*N + n0 + int64_t(sg_id)*TN),
                    N, mat::layout::row_major);
        });
    });
}

// Fused gate+up expert GEMM.
//
// gate and up are separate [H -> EF] matrices applied to the SAME activation
// tile, and were two launches: A staged into SLM twice and read from SLM once
// per MAC each time. Fusing them stages A once and reads it once per TWO MACs,
// cutting loads-per-MAC from (TM+TN)/(TM*TN) to (TM+2*TN)/(2*TM*TN).
template <int BM,int BN,int BK,int TM,int TN, bool BIGGRF = false>
static sycl::event k_iq2kt_moe_gemm2(sycl::queue& q, const int8_t* ttab,
        uint8_t* const* bases, const int32_t* eoff, const int32_t* ecnt,
        const int32_t* goff, const int32_t* tiles, int ntiles,
        int64_t woff_g, int64_t woff_u,
        const int32_t* tokall, const float* xg, int64_t x_stride,
        float* og, float* ou, int64_t K, int64_t N,
        const std::vector<sycl::event>& deps) {
    constexpr int MT = BM/TM, NT = BN/TN, WG = MT*NT;
    constexpr int PA = BM + 1, PB = BN + 1;
    const int64_t nb = K/256, row_bytes = 4 + nb*68;
    const int64_t ntn = (N + BN - 1)/BN;
    (void)ttab;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float,1> As (sycl::range<1>(size_t(BK)*PA), h);
        sycl::local_accessor<float,1> Bg (sycl::range<1>(size_t(BK)*PB), h);
        sycl::local_accessor<float,1> Bu (sycl::range<1>(size_t(BK)*PB), h);
        auto body = [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int64_t gid = it.get_group(0);
            const int ti = int(gid / ntn);
            const int64_t n0 = (gid % ntn)*BN;
            const int ke = tiles[2*ti], m0 = tiles[2*ti + 1];
            const int nt = ecnt[ke], eo = eoff[ke], go = goff[ke];
            const uint8_t* wg_ = bases[ke] + woff_g;
            const uint8_t* wu_ = bases[ke] + woff_u;
            const int lid = int(it.get_local_id(0));
            const int lm = lid % MT, ln = lid / MT;
            float ag[TM][TN], au[TM][TN];
#pragma unroll
            for (int i=0;i<TM;++i)
#pragma unroll
                for (int j=0;j<TN;++j) { ag[i][j] = 0.f; au[i][j] = 0.f; }
            for (int64_t k0 = 0; k0 < K; k0 += BK) {
                it.barrier(sycl::access::fence_space::local_space);
                for (int idx = lid; idx < BM*BK; idx += WG) {
                    const int mm = idx / BK, kk = idx - mm*BK;
                    const int m = m0 + mm;
                    float v = 0.f;
                    if (m < nt) {
                        const int64_t srow = tokall ? int64_t(tokall[eo + m])
                                                    : int64_t(go + m);
                        v = xg[srow*x_stride + k0 + kk];
                    }
                    As[size_t(kk)*PA + mm] = v;
                }
                for (int idx = lid; idx < BN*(BK/8); idx += WG) {
                    const int gg = idx / BN, nn = idx - gg*BN;
                    const int64_t row = n0 + nn;
                    if (row >= N) {
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            Bg[size_t(gg*8+j)*PB + nn] = 0.f;
                            Bu[size_t(gg*8+j)*PB + nn] = 0.f;
                        }
                        continue;
                    }
                    const int64_t gI = (k0 >> 3) + gg;
                    const int ib = int(gI & 31);
#pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        const uint8_t* rp = (half ? wu_ : wg_) + row*row_bytes;
                        float scale; std::memcpy(&scale, rp, 4);
                        const uint8_t* b = rp + 4 + (gI >> 5)*68;
                        const uint16_t ql16 = uint16_t(b[4+2*ib]) | (uint16_t(b[4+2*ib+1])<<8);
                        const uint32_t seed = uint32_t(ql16) + 4096u;
                        const int sc = (b[(ib/4)%4] >> (4*(ib/16))) & 0xf;
                        const float dl = scale * float(kIq4kValues[sc]) * 1.05f;
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            const float v = dl * float(trellis_at(seed, j));
                            if (half) Bu[size_t(gg*8+j)*PB + nn] = v;
                            else      Bg[size_t(gg*8+j)*PB + nn] = v;
                        }
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
#pragma unroll 4
                for (int kk = 0; kk < BK; ++kk) {
                    float a[TM], bg[TN], bu[TN];
#pragma unroll
                    for (int i=0;i<TM;++i) a[i] = As[size_t(kk)*PA + lm*TM + i];
#pragma unroll
                    for (int j=0;j<TN;++j) {
                        bg[j] = Bg[size_t(kk)*PB + ln*TN + j];
                        bu[j] = Bu[size_t(kk)*PB + ln*TN + j];
                    }
#pragma unroll
                    for (int i=0;i<TM;++i)
#pragma unroll
                        for (int j=0;j<TN;++j) {
                            ag[i][j] += a[i]*bg[j];
                            au[i][j] += a[i]*bu[j];
                        }
                }
            }
#pragma unroll
            for (int i=0;i<TM;++i) {
                const int m = m0 + lm*TM + i;
                if (m >= nt) continue;
#pragma unroll
                for (int j=0;j<TN;++j) {
                    const int64_t col = n0 + ln*TN + j;
                    if (col < N) {
                        og[int64_t(go + m)*N + col] = ag[i][j];
                        ou[int64_t(go + m)*N + col] = au[i][j];
                    }
                }
            }
        };
        const sycl::nd_range<1> nr(size_t(ntiles)*size_t(ntn)*WG, WG);
        if constexpr (BIGGRF)
            h.parallel_for(nr, sycl::ext::oneapi::experimental::properties{
                               sycl::ext::intel::experimental::grf_size<256>}, body);
        else h.parallel_for(nr, body);
    });
}

template <int BM,int BN,int BK,int TM,int TN, bool BIGGRF = false>
static sycl::event k_iq2kt_moe_gemm(sycl::queue& q, const int8_t* ttab,
        uint8_t* const* bases, const int32_t* eoff, const int32_t* ecnt,
        const int32_t* goff, const int32_t* tiles, int ntiles, int64_t woff,
        const int32_t* tokall, const float* xg, int64_t x_stride,
        float* og, int64_t K, int64_t N,
        const std::vector<sycl::event>& deps) {
    constexpr int MT = BM/TM, NT = BN/TN, WG = MT*NT;
    constexpr int PA = BM + 1, PB = BN + 1;
    const int64_t nb = K/256, row_bytes = 4 + nb*68;
    const int64_t ntn = (N + BN - 1)/BN;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float,1> As(sycl::range<1>(size_t(BK)*PA), h);
        sycl::local_accessor<float,1> Bs(sycl::range<1>(size_t(BK)*PB), h);
        auto body = [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int64_t gid = it.get_group(0);
            const int ti = int(gid / ntn);
            const int64_t n0 = (gid % ntn)*BN;
            const int ke = tiles[2*ti], m0 = tiles[2*ti + 1];
            const uint8_t* wb = bases[ke] + woff;
            const int nt = ecnt[ke], eo = eoff[ke], go = goff[ke];
            const int lid = int(it.get_local_id(0));
            const int lm = lid % MT, ln = lid / MT;
            float acc[TM][TN];
#pragma unroll
            for (int i=0;i<TM;++i)
#pragma unroll
                for (int j=0;j<TN;++j) acc[i][j] = 0.f;
            for (int64_t k0 = 0; k0 < K; k0 += BK) {
                it.barrier(sycl::access::fence_space::local_space);
                for (int idx = lid; idx < BM*BK; idx += WG) {
                    const int mm = idx / BK, kk = idx - mm*BK;
                    const int m = m0 + mm;
                    float v = 0.f;
                    if (m < nt) {
                        const int64_t srow = tokall ? int64_t(tokall[eo + m])
                                                    : int64_t(go + m);
                        v = xg[srow*x_stride + k0 + kk];
                    }
                    As[size_t(kk)*PA + mm] = v;
                }
                for (int idx = lid; idx < BN*(BK/8); idx += WG) {
                    const int gg = idx / BN, nn = idx - gg*BN;
                    const int64_t row = n0 + nn;
                    if (row >= N) {
#pragma unroll
                        for (int j = 0; j < 8; ++j) Bs[size_t(gg*8+j)*PB + nn] = 0.f;
                        continue;
                    }
                    const uint8_t* rp = wb + row*row_bytes;
                    float scale; std::memcpy(&scale, rp, 4);
                    const int64_t gI = (k0 >> 3) + gg;
                    const uint8_t* b = rp + 4 + (gI >> 5)*68;
                    const int ib = int(gI & 31);
                    const uint16_t ql16 = uint16_t(b[4+2*ib]) | (uint16_t(b[4+2*ib+1])<<8);
                    const uint32_t seed = uint32_t(ql16) + 4096u;
                    const int sc = (b[(ib/4)%4] >> (4*(ib/16))) & 0xf;
                    const float dl = scale * float(kIq4kValues[sc]) * 1.05f;
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        Bs[size_t(gg*8+j)*PB + nn] = dl * float(trellis_at(seed, j));
                }
                it.barrier(sycl::access::fence_space::local_space);
#pragma unroll 4
                for (int kk = 0; kk < BK; ++kk) {
                    float a[TM], b[TN];
#pragma unroll
                    for (int i=0;i<TM;++i) a[i] = As[size_t(kk)*PA + lm*TM + i];
#pragma unroll
                    for (int j=0;j<TN;++j) b[j] = Bs[size_t(kk)*PB + ln*TN + j];
#pragma unroll
                    for (int i=0;i<TM;++i)
#pragma unroll
                        for (int j=0;j<TN;++j) acc[i][j] += a[i]*b[j];
                }
            }
#pragma unroll
            for (int i=0;i<TM;++i) {
                const int m = m0 + lm*TM + i;
                if (m >= nt) continue;
#pragma unroll
                for (int j=0;j<TN;++j) {
                    const int64_t col = n0 + ln*TN + j;
                    if (col < N) og[int64_t(go + m)*N + col] = acc[i][j];
                }
            }
        };
        const sycl::nd_range<1> nr(size_t(ntiles)*size_t(ntn)*WG, WG);
        if constexpr (BIGGRF)
            h.parallel_for(nr, sycl::ext::oneapi::experimental::properties{
                               sycl::ext::intel::experimental::grf_size<256>}, body);
        else h.parallel_for(nr, body);
    });
}

static sycl::event k_q6_gemm(sycl::queue& q, const uint8_t* w, const float* x,
                             float* y, int64_t K, int64_t N, int64_t row_bytes,
                             int T, const std::vector<sycl::event>& deps = {}) {
    if (g_gemm_v2 && T >= 32 && (K % 32) == 0) {
        if (N >= 1024 && T >= 128)
            return k_q6_gemm_v3<128,256,32,8,8,true>(q, w, x, y, K, N, row_bytes, T, deps);
        if (N >= 1024) return k_q6_gemm_v3<64,256,32,4,4>(q, w, x, y, K, N, row_bytes, T, deps);
        return k_q6_gemm_v3<64,64,32,4,4>(q, w, x, y, K, N, row_bytes, T, deps);
    }
    return k_q6_gemm_v1(q, w, x, y, K, N, row_bytes, T, deps);
}

static sycl::event k_iq2kt_gemm(sycl::queue& q, const uint8_t* base,
                                const float* x, int64_t x_stride,
                                const int32_t* tok, int nt,
                                float* out, int64_t K, int64_t N,
                                const std::vector<sycl::event>& deps) {
    return k_iq2kt_gemm_v1(q, base, x, x_stride, tok, nt, out, K, N, deps);
}

// RoPE, LLAMA_ROPE_TYPE_NORM: rotate ADJACENT pairs (v[2i], v[2i+1]).
// Applied over n_rot dims of each of `nh` heads with stride `stride`.
static sycl::event k_rope_norm(sycl::queue& q, float* v, int nh, int64_t stride,
                               int n_rot, int pos, float theta_base) {
    return q.parallel_for(sycl::range<1>(size_t(nh)*(n_rot/2)), [=](sycl::id<1> id) {
        const int i = int(id[0]);
        const int head = i / (n_rot/2);
        const int k = i % (n_rot/2);
        const float inv = sycl::pow(theta_base, -float(2*k)/float(n_rot));
        const float ang = float(pos)*inv;
        const float c = sycl::cos(ang), s = sycl::sin(ang);
        float* p = v + head*stride + 2*k;
        const float a = p[0], b = p[1];
        p[0] = a*c - b*s;
        p[1] = a*s + b*c;
    });
}

// MLA scores: per head, score[t] = qc[h].ckv[t] + qr[h].kr[t], scaled, softmaxed,
// then o_c[h] = sum_t a[t]*ckv[t].  One work-group per head.
// Decode-path MLA with HG heads per work-group. Same argument as the prefill
// kernel: one latent KV cache shared by 64 heads, but a work-group per head read
// it 64 times -- ~43 GB per token across 78 blocks, which is why decode
// attention measured 98 ms/token.
template <int HG>
static sycl::event k_mla_attend_hg(sycl::queue& q, const float* qc, const float* qr,
                                   const float* ckv, const float* kr, float* oc,
                                   int nh, int n_kv, int kvl, int rl, float scale,
                                   float* scratch) {
    constexpr int WG = 256, NSG = WG/32;
    const int qd = kvl + rl, ngh = nh / HG;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float,1> qs  (sycl::range<1>(size_t(HG)*qd), h);
        sycl::local_accessor<float,1> redm(sycl::range<1>(size_t(HG)*NSG), h);
        sycl::local_accessor<float,1> redl(sycl::range<1>(size_t(HG)*NSG), h);
        auto body = [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int head0 = int(it.get_group(0))*HG;
            const int tid = int(it.get_local_id(0));
            auto sg = it.get_sub_group();
            const int sgid = int(sg.get_group_id()[0]), lane = int(sg.get_local_id()[0]);
            for (int idx = tid; idx < HG*qd; idx += WG) {
                const int hh = idx/qd, jj = idx - hh*qd;
                qs[idx] = (jj < kvl) ? qc[size_t(head0+hh)*kvl + jj]
                                     : qr[size_t(head0+hh)*rl + (jj-kvl)];
            }
            it.barrier(sycl::access::fence_space::local_space);
            for (int t = tid; t < n_kv; t += WG) {
                const float* c = ckv + size_t(t)*kvl;
                const float* r = kr  + size_t(t)*rl;
                float sv[HG];
#pragma unroll
                for (int hh = 0; hh < HG; ++hh) sv[hh] = 0.f;
                for (int j = 0; j < kvl; ++j) { const float cj = c[j];
#pragma unroll
                    for (int hh = 0; hh < HG; ++hh) sv[hh] += qs[size_t(hh)*qd + j]*cj; }
                for (int j = 0; j < rl; ++j) { const float rj = r[j];
#pragma unroll
                    for (int hh = 0; hh < HG; ++hh) sv[hh] += qs[size_t(hh)*qd + kvl + j]*rj; }
#pragma unroll
                for (int hh = 0; hh < HG; ++hh)
                    scratch[size_t(head0+hh)*n_kv + t] = sv[hh]*scale;
            }
            it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
            for (int hh = 0; hh < HG; ++hh) {
                float m = -INFINITY;
                for (int t = tid; t < n_kv; t += WG)
                    m = sycl::max(m, scratch[size_t(head0+hh)*n_kv + t]);
                m = sycl::reduce_over_group(sg, m, sycl::maximum<float>());
                if (lane == 0) redm[size_t(hh)*NSG + sgid] = m;
            }
            it.barrier(sycl::access::fence_space::local_space);
            float mv[HG], lv[HG];
#pragma unroll
            for (int hh = 0; hh < HG; ++hh) {
                float m = -INFINITY;
                for (int i = 0; i < NSG; ++i) m = sycl::max(m, redm[size_t(hh)*NSG + i]);
                mv[hh] = m;
                float s2 = 0.f;
                for (int t = tid; t < n_kv; t += WG) {
                    const float e = sycl::exp(scratch[size_t(head0+hh)*n_kv + t] - m);
                    scratch[size_t(head0+hh)*n_kv + t] = e;
                    s2 += e;
                }
                s2 = sycl::reduce_over_group(sg, s2, sycl::plus<float>());
                if (lane == 0) redl[size_t(hh)*NSG + sgid] = s2;
            }
            it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
            for (int hh = 0; hh < HG; ++hh) {
                float s2 = 0.f;
                for (int i = 0; i < NSG; ++i) s2 += redl[size_t(hh)*NSG + i];
                lv[hh] = 1.f/s2;
            }
            for (int d = tid; d < kvl; d += WG) {
                float a[HG];
#pragma unroll
                for (int hh = 0; hh < HG; ++hh) a[hh] = 0.f;
                for (int t = 0; t < n_kv; ++t) {
                    const float cd = ckv[size_t(t)*kvl + d];
#pragma unroll
                    for (int hh = 0; hh < HG; ++hh)
                        a[hh] += scratch[size_t(head0+hh)*n_kv + t]*cd;
                }
#pragma unroll
                for (int hh = 0; hh < HG; ++hh)
                    oc[size_t(head0+hh)*kvl + d] = a[hh]*lv[hh];
            }
        };
        h.parallel_for(sycl::nd_range<1>(size_t(ngh)*WG, WG),
                       sycl::ext::oneapi::experimental::properties{
                           sycl::ext::intel::experimental::grf_size<256>}, body);
    });
}

static sycl::event k_mla_attend(sycl::queue& q, const float* qc, const float* qr,
                                const float* ckv, const float* kr, float* oc,
                                int nh, int n_kv, int kvl, int rl, float scale,
                                float* scratch) {
    constexpr int WG = 256;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float,1> red(sycl::range<1>(WG/32), h);
        h.parallel_for(sycl::nd_range<1>(size_t(nh)*WG, WG), [=](sycl::nd_item<1> it)
                       [[sycl::reqd_sub_group_size(32)]] {
            const int head = int(it.get_group(0));
            const int tid = int(it.get_local_id(0));
            const float* qch = qc + size_t(head)*kvl;
            const float* qrh = qr + size_t(head)*rl;
            float* sc = scratch + size_t(head)*n_kv;

            for (int t = tid; t < n_kv; t += WG) {
                const float* c = ckv + size_t(t)*kvl;
                const float* r = kr  + size_t(t)*rl;
                float s = 0.f;
                for (int j = 0; j < kvl; ++j) s += qch[j]*c[j];
                for (int j = 0; j < rl;  ++j) s += qrh[j]*r[j];
                sc[t] = s*scale;
            }
            it.barrier(sycl::access::fence_space::local_space);

            float m = -INFINITY;
            for (int t = tid; t < n_kv; t += WG) m = sycl::max(m, sc[t]);
            auto sg = it.get_sub_group();
            m = sycl::reduce_over_group(sg, m, sycl::maximum<float>());
            if (sg.get_local_id()[0]==0) red[sg.get_group_id()[0]] = m;
            it.barrier(sycl::access::fence_space::local_space);
            for (int i = 0; i < WG/32; ++i) m = sycl::max(m, red[i]);
            it.barrier(sycl::access::fence_space::local_space);

            float sum = 0.f;
            for (int t = tid; t < n_kv; t += WG) { float e = sycl::exp(sc[t]-m); sc[t]=e; sum+=e; }
            sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
            if (sg.get_local_id()[0]==0) red[sg.get_group_id()[0]] = sum;
            it.barrier(sycl::access::fence_space::local_space);
            float tot = 0.f;
            for (int i = 0; i < WG/32; ++i) tot += red[i];
            const float inv = 1.f/tot;
            it.barrier(sycl::access::fence_space::local_space);

            float* och = oc + size_t(head)*kvl;
            for (int j = tid; j < kvl; j += WG) {
                float a = 0.f;
                for (int t = 0; t < n_kv; ++t) a += sc[t]*ckv[size_t(t)*kvl + j];
                och[j] = a*inv;
            }
        });
    });
}

static sycl::event k_silu_mul(sycl::queue& q, const float* g, const float* u,
                              float* y, int64_t n) {
    return q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> i) {
        const float x = g[i];
        y[i] = (x/(1.f+sycl::exp(-x)))*u[i];
    });
}
static sycl::event k_add(sycl::queue& q, float* a, const float* b, int64_t n) {
    return q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> i){ a[i]+=b[i]; });
}
static sycl::event k_axpy(sycl::queue& q, float* a, const float* b, float w, int64_t n) {
    return q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> i){ a[i]+=w*b[i]; });
}
static sycl::event k_zero(sycl::queue& q, float* a, int64_t n) {
    return q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> i){ a[i]=0.f; });
}

// ---- batched variants for prefill --------------------------------------
// One work-group per token row; identical arithmetic to the single-token
// kernels, so prefill and decode stay numerically consistent.
static sycl::event k_rmsnorm_T(sycl::queue& q, const float* x, const float* w,
                               float* y, int64_t n, float eps, int T) {
    constexpr int WG = 256;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float,1> red(sycl::range<1>(WG/32), h);
        h.parallel_for(sycl::nd_range<1>(size_t(T)*WG, WG), [=](sycl::nd_item<1> it)
                       [[sycl::reqd_sub_group_size(32)]] {
            const int64_t t = it.get_group(0);
            const int tid = int(it.get_local_id(0));
            const float* xr = x + t*n;
            float* yr = y + t*n;
            float s = 0.f;
            for (int64_t i = tid; i < n; i += WG) s += xr[i]*xr[i];
            auto sg = it.get_sub_group();
            s = sycl::reduce_over_group(sg, s, sycl::plus<float>());
            if (sg.get_local_id()[0]==0) red[sg.get_group_id()[0]] = s;
            it.barrier(sycl::access::fence_space::local_space);
            float tot = 0.f;
            for (int i = 0; i < WG/32; ++i) tot += red[i];
            const float inv = sycl::rsqrt(tot/float(n) + eps);
            for (int64_t i = tid; i < n; i += WG) yr[i] = xr[i]*inv*w[i];
        });
    });
}

static sycl::event k_rope_norm_T(sycl::queue& q, float* v, int nh, int64_t stride,
                                 int n_rot, int pos0, float theta_base,
                                 int T, int64_t tok_stride) {
    return q.parallel_for(sycl::range<1>(size_t(T)*nh*(n_rot/2)), [=](sycl::id<1> id) {
        const int i = int(id[0]);
        const int per = nh*(n_rot/2);
        const int t = i / per;
        const int r = i % per;
        const int head = r / (n_rot/2);
        const int k = r % (n_rot/2);
        const float inv = sycl::pow(theta_base, -float(2*k)/float(n_rot));
        const float ang = float(pos0 + t)*inv;
        const float c = sycl::cos(ang), s = sycl::sin(ang);
        float* p = v + int64_t(t)*tok_stride + int64_t(head)*stride + 2*k;
        const float a = p[0], b = p[1];
        p[0] = a*c - b*s;
        p[1] = a*s + b*c;
    });
}

static sycl::event k_silu_mul_T(sycl::queue& q, const float* g, const float* u,
                                float* y, int64_t n) {
    return q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> i) {
        const float x = g[i];
        y[i] = (x/(1.f+sycl::exp(-x)))*u[i];
    });
}

// Causal MLA over T queries. Work-group per (token, head); query at absolute
// position pos0+t attends to cache positions [0, pos0+t].
static sycl::event k_mla_attend_T(sycl::queue& q, const float* qc, const float* qr,
                                  const float* ckv, const float* kr, float* oc,
                                  int nh, int kvl, int rl, float scale,
                                  float* scratch, int T, int pos0, int n_ctx) {
    constexpr int WG = 128;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float,1> red(sycl::range<1>(WG/32), h);
        h.parallel_for(sycl::nd_range<1>(size_t(T)*nh*WG, WG), [=](sycl::nd_item<1> it)
                       [[sycl::reqd_sub_group_size(32)]] {
            const int64_t wg = it.get_group(0);
            const int t = int(wg / nh);
            const int head = int(wg % nh);
            const int tid = int(it.get_local_id(0));
            const int n_kv = pos0 + t + 1;                 // causal
            const float* qch = qc + (int64_t(t)*nh + head)*kvl;
            const float* qrh = qr + (int64_t(t)*nh + head)*rl;
            float* sc = scratch + (int64_t(t)*nh + head)*int64_t(n_ctx);

            for (int p = tid; p < n_kv; p += WG) {
                const float* c = ckv + int64_t(p)*kvl;
                const float* r = kr  + int64_t(p)*rl;
                float v = 0.f;
                for (int j = 0; j < kvl; ++j) v += qch[j]*c[j];
                for (int j = 0; j < rl;  ++j) v += qrh[j]*r[j];
                sc[p] = v*scale;
            }
            it.barrier(sycl::access::fence_space::local_space);
            float m = -INFINITY;
            for (int p = tid; p < n_kv; p += WG) m = sycl::max(m, sc[p]);
            auto sg = it.get_sub_group();
            m = sycl::reduce_over_group(sg, m, sycl::maximum<float>());
            if (sg.get_local_id()[0]==0) red[sg.get_group_id()[0]] = m;
            it.barrier(sycl::access::fence_space::local_space);
            for (int i = 0; i < WG/32; ++i) m = sycl::max(m, red[i]);
            it.barrier(sycl::access::fence_space::local_space);
            float sum = 0.f;
            for (int p = tid; p < n_kv; p += WG) { float e = sycl::exp(sc[p]-m); sc[p]=e; sum+=e; }
            sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
            if (sg.get_local_id()[0]==0) red[sg.get_group_id()[0]] = sum;
            it.barrier(sycl::access::fence_space::local_space);
            float tot = 0.f;
            for (int i = 0; i < WG/32; ++i) tot += red[i];
            const float inv = 1.f/tot;
            it.barrier(sycl::access::fence_space::local_space);
            float* och = oc + (int64_t(t)*nh + head)*kvl;
            for (int j = tid; j < kvl; j += WG) {
                float a = 0.f;
                for (int p = 0; p < n_kv; ++p) a += sc[p]*ckv[int64_t(p)*kvl + j];
                och[j] = a*inv;
            }
        });
    });
}

// Flash-style causal MLA: streaming softmax, NO materialised score buffer.
//
// The tiled version needed T x heads x n_ctx floats of scratch — 70 GB for a
// 512-token chunk — which capped the prefill chunk at 64 and forced the expert
// bank to be re-read once per chunk. Here the running max/sum and the 512-wide
// accumulator live in local memory, so chunk size is bounded by nothing but the
// KV cache, and the whole prompt can be one batch.
// Flash MLA, HG heads per work-group.
//
// MLA keeps ONE latent KV cache shared by all 64 heads, but the previous kernel
// gave each (token, head) its own work-group, so the whole cache was re-read 64
// times per token: at a 2000-token chunk that is ~295 GB of reads per block for
// 21.7 TFLOP of work, and it ran at 750 GF/s -- one global load per MAC.
//
// Here a work-group owns HG heads of one token. Their q vectors are staged in
// SLM once, and each KV element loaded from memory feeds HG MACs, so both the
// score pass and the value pass drop to 1/HG loads per MAC and the cache is read
// 64/HG times instead of 64. The streaming-softmax maths is unchanged.
// Flash MLA with a 2D register tile.
//
// Head grouping (k_mla_flash_TH) cut GLOBAL KV traffic 8x but left ~1.1 loads
// per MAC inside the tile: each lane still read HG q values from SLM for HG
// MACs. Position tiling on top of that spilled, because m_new[HG]/corr[HG]/a[HG]
// already pinned 24 registers.
//
// Here the work-group's lanes are a (head-column x position) grid: a lane owns
// TM heads and TN KV positions, so one q read feeds TN MACs and one K read feeds
// TM -- (TM+TN) loads per TM*TN MACs, 0.5 at 4x4 against 1.125. Splitting heads
// across MG columns also shrinks the per-lane softmax state from HG to TM.
// Flash MLA with REGISTER-RESIDENT output accumulators.
//
// k_mla_flash_RT is VRAM-bandwidth-bound: with HG=8 the shared latent cache is
// still read 8 times per token, ~5.4 TB per prefill pass, and it runs at ~90% of
// measured VRAM bandwidth. Halving that means HG=16 -- which RT cannot do,
// because its per-work-group SLM (q tile 36.9 KB + accumulators 32.8 KB) kills
// occupancy.
//
// The accumulators do not need to be in SLM at all. Each lane can own a fixed
// (all HG heads) x TD dims slice for the whole KV loop and keep it in registers,
// which removes HG*kvl floats of SLM and makes HG=16 fit. Requires WG*TD == kvl.
template <int HG, int WG, int TD>
static sycl::event k_mla_flash_RG(sycl::queue& q, const float* qc, const float* qr,
                                  const float* ckv, const float* kr, float* oc,
                                  int nh, int kvl, int rl, float scale,
                                  int T, int pos0) {
    constexpr int NSG = WG/32;
    const int ngh = nh / HG, qd = kvl + rl;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float,1> qs   (sycl::range<1>(size_t(HG)*qd), h);
        sycl::local_accessor<float,1> wls  (sycl::range<1>(size_t(HG)*WG), h);
        sycl::local_accessor<float,1> redm (sycl::range<1>(size_t(HG)*NSG), h);
        sycl::local_accessor<float,1> redl (sycl::range<1>(size_t(HG)*NSG), h);
        sycl::local_accessor<float,1> mst  (sycl::range<1>(HG), h);
        sycl::local_accessor<float,1> lst  (sycl::range<1>(HG), h);
        sycl::local_accessor<float,1> corrs(sycl::range<1>(HG), h);
        sycl::local_accessor<float,1> mnews(sycl::range<1>(HG), h);
        auto body = [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int64_t wg = it.get_group(0);
            const int t = int(wg / ngh);
            const int head0 = int(wg % ngh)*HG;
            const int tid = int(it.get_local_id(0));
            const int n_kv = pos0 + t + 1;
            const int d0 = tid*TD;                 // this lane's permanent dims
            auto sg = it.get_sub_group();
            const int sgid = int(sg.get_group_id()[0]), lane = int(sg.get_local_id()[0]);

            for (int idx = tid; idx < HG*qd; idx += WG) {
                const int hh = idx / qd, jj = idx - hh*qd;
                qs[idx] = (jj < kvl) ? qc[(int64_t(t)*nh + head0 + hh)*kvl + jj]
                                     : qr[(int64_t(t)*nh + head0 + hh)*rl + (jj - kvl)];
            }
            if (tid < HG) { mst[tid] = -INFINITY; lst[tid] = 0.f; }
            float a[HG][TD];
#pragma unroll
            for (int hh = 0; hh < HG; ++hh)
#pragma unroll
                for (int e = 0; e < TD; ++e) a[hh][e] = 0.f;
            it.barrier(sycl::access::fence_space::local_space);

            for (int p0 = 0; p0 < n_kv; p0 += WG) {
                const int p = p0 + tid;
                float v[HG];
#pragma unroll
                for (int hh = 0; hh < HG; ++hh) v[hh] = 0.f;
                if (p < n_kv) {
                    const float* c = ckv + int64_t(p)*kvl;
                    const float* r = kr  + int64_t(p)*rl;
                    for (int j = 0; j < kvl; ++j) { const float cj = c[j];
#pragma unroll
                        for (int hh = 0; hh < HG; ++hh) v[hh] += qs[size_t(hh)*qd + j]*cj; }
                    for (int j = 0; j < rl; ++j) { const float rj = r[j];
#pragma unroll
                        for (int hh = 0; hh < HG; ++hh) v[hh] += qs[size_t(hh)*qd + kvl + j]*rj; }
#pragma unroll
                    for (int hh = 0; hh < HG; ++hh) v[hh] *= scale;
                } else {
#pragma unroll
                    for (int hh = 0; hh < HG; ++hh) v[hh] = -INFINITY;
                }
#pragma unroll
                for (int hh = 0; hh < HG; ++hh) {
                    const float mx = sycl::reduce_over_group(sg, v[hh], sycl::maximum<float>());
                    if (lane == 0) redm[size_t(hh)*NSG + sgid] = mx;
                }
                it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                for (int hh = 0; hh < HG; ++hh) {
                    float m_tile = -INFINITY;
                    for (int i = 0; i < NSG; ++i)
                        m_tile = sycl::max(m_tile, redm[size_t(hh)*NSG + i]);
                    const float m_old = mst[hh];
                    const float mn = sycl::max(m_old, m_tile);
                    const float w = (p < n_kv) ? sycl::exp(v[hh] - mn) : 0.f;
                    wls[size_t(hh)*WG + tid] = w;
                    const float ls = sycl::reduce_over_group(sg, w, sycl::plus<float>());
                    if (lane == 0) redl[size_t(hh)*NSG + sgid] = ls;
                    if (tid == 0) {
                        corrs[hh] = (m_old == -INFINITY) ? 0.f : sycl::exp(m_old - mn);
                        mnews[hh] = mn;
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
                if (tid == 0)
#pragma unroll
                    for (int hh = 0; hh < HG; ++hh) {
                        float l_tile = 0.f;
                        for (int i = 0; i < NSG; ++i) l_tile += redl[size_t(hh)*NSG + i];
                        lst[hh] = lst[hh]*corrs[hh] + l_tile;
                        mst[hh] = mnews[hh];
                    }
                it.barrier(sycl::access::fence_space::local_space);
                const int nend = sycl::min(WG, n_kv - p0);
#pragma unroll
                for (int hh = 0; hh < HG; ++hh) {
                    const float cr = corrs[hh];
#pragma unroll
                    for (int e = 0; e < TD; ++e) a[hh][e] *= cr;
                }
                for (int j = 0; j < nend; ++j) {
                    float cd[TD];
#pragma unroll
                    for (int e = 0; e < TD; ++e) cd[e] = ckv[int64_t(p0 + j)*kvl + d0 + e];
#pragma unroll
                    for (int hh = 0; hh < HG; ++hh) {
                        const float w = wls[size_t(hh)*WG + j];
#pragma unroll
                        for (int e = 0; e < TD; ++e) a[hh][e] += w*cd[e];
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
            }
#pragma unroll
            for (int hh = 0; hh < HG; ++hh) {
                const float inv = 1.f/lst[hh];
#pragma unroll
                for (int e = 0; e < TD; ++e)
                    oc[(int64_t(t)*nh + head0 + hh)*kvl + d0 + e] = a[hh][e]*inv;
            }
        };
        h.parallel_for(sycl::nd_range<1>(size_t(T)*ngh*WG, WG),
                       sycl::ext::oneapi::experimental::properties{
                           sycl::ext::intel::experimental::grf_size<256>}, body);
    });
}

template <int HG, int WG, int TM, int TN, int TD>
static sycl::event k_mla_flash_RT(sycl::queue& q, const float* qc, const float* qr,
                                  const float* ckv, const float* kr, float* oc,
                                  int nh, int kvl, int rl, float scale,
                                  int T, int pos0) {
    constexpr int MG   = HG/TM;          // head columns
    constexpr int NLC  = WG/MG;          // lanes per head column
    constexpr int TW   = NLC*TN;         // KV positions per tile
    constexpr int NSG  = WG/32;
    constexpr int NSGC = NSG/MG;         // sub-groups per head column
    const int ngh = nh / HG, qd = kvl + rl;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float,1> qs   (sycl::range<1>(size_t(HG)*qd), h);
        sycl::local_accessor<float,1> accs (sycl::range<1>(size_t(HG)*kvl), h);
        sycl::local_accessor<float,1> wls  (sycl::range<1>(size_t(HG)*TW), h);
        sycl::local_accessor<float,1> redm (sycl::range<1>(size_t(HG)*NSG), h);
        sycl::local_accessor<float,1> redl (sycl::range<1>(size_t(HG)*NSG), h);
        sycl::local_accessor<float,1> mst  (sycl::range<1>(HG), h);
        sycl::local_accessor<float,1> lst  (sycl::range<1>(HG), h);
        sycl::local_accessor<float,1> corrs(sycl::range<1>(HG), h);
        sycl::local_accessor<float,1> mnews(sycl::range<1>(HG), h);
        auto body = [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int64_t wg = it.get_group(0);
            const int t = int(wg / ngh);
            const int head0 = int(wg % ngh)*HG;
            const int tid = int(it.get_local_id(0));
            const int m_idx = tid / NLC, n_idx = tid % NLC, hb = m_idx*TM;
            const int n_kv = pos0 + t + 1;
            auto sg = it.get_sub_group();
            const int sgid = int(sg.get_group_id()[0]), lane = int(sg.get_local_id()[0]);

            for (int idx = tid; idx < HG*qd; idx += WG) {
                const int hh = idx / qd, jj = idx - hh*qd;
                qs[idx] = (jj < kvl) ? qc[(int64_t(t)*nh + head0 + hh)*kvl + jj]
                                     : qr[(int64_t(t)*nh + head0 + hh)*rl + (jj - kvl)];
            }
            for (int idx = tid; idx < HG*kvl; idx += WG) accs[idx] = 0.f;
            if (tid < HG) { mst[tid] = -INFINITY; lst[tid] = 0.f; }
            it.barrier(sycl::access::fence_space::local_space);

            for (int p0 = 0; p0 < n_kv; p0 += TW) {
                const int pb = p0 + n_idx*TN;
                float v[TM][TN];
#pragma unroll
                for (int i = 0; i < TM; ++i)
#pragma unroll
                    for (int k = 0; k < TN; ++k) v[i][k] = 0.f;
                for (int j = 0; j < kvl; ++j) {
                    float kk[TN];
#pragma unroll
                    for (int k = 0; k < TN; ++k)
                        kk[k] = (pb + k < n_kv) ? ckv[int64_t(pb + k)*kvl + j] : 0.f;
#pragma unroll
                    for (int i = 0; i < TM; ++i) {
                        const float qv = qs[size_t(hb + i)*qd + j];
#pragma unroll
                        for (int k = 0; k < TN; ++k) v[i][k] += qv*kk[k];
                    }
                }
                for (int j = 0; j < rl; ++j) {
                    float kk[TN];
#pragma unroll
                    for (int k = 0; k < TN; ++k)
                        kk[k] = (pb + k < n_kv) ? kr[int64_t(pb + k)*rl + j] : 0.f;
#pragma unroll
                    for (int i = 0; i < TM; ++i) {
                        const float qv = qs[size_t(hb + i)*qd + kvl + j];
#pragma unroll
                        for (int k = 0; k < TN; ++k) v[i][k] += qv*kk[k];
                    }
                }
#pragma unroll
                for (int i = 0; i < TM; ++i)
#pragma unroll
                    for (int k = 0; k < TN; ++k)
                        v[i][k] = (pb + k < n_kv) ? v[i][k]*scale : -INFINITY;

#pragma unroll
                for (int i = 0; i < TM; ++i) {
                    float lm = -INFINITY;
#pragma unroll
                    for (int k = 0; k < TN; ++k) lm = sycl::max(lm, v[i][k]);
                    lm = sycl::reduce_over_group(sg, lm, sycl::maximum<float>());
                    if (lane == 0) redm[size_t(hb + i)*NSG + sgid] = lm;
                }
                it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                for (int i = 0; i < TM; ++i) {
                    const int hh = hb + i;
                    float m_tile = -INFINITY;
#pragma unroll
                    for (int c = 0; c < NSGC; ++c)
                        m_tile = sycl::max(m_tile, redm[size_t(hh)*NSG + m_idx*NSGC + c]);
                    const float m_old = mst[hh];
                    const float mn = sycl::max(m_old, m_tile);
                    const float cr = (m_old == -INFINITY) ? 0.f : sycl::exp(m_old - mn);
                    float ls = 0.f;
#pragma unroll
                    for (int k = 0; k < TN; ++k) {
                        const float w = (pb + k < n_kv) ? sycl::exp(v[i][k] - mn) : 0.f;
                        wls[size_t(hh)*TW + n_idx*TN + k] = w;
                        ls += w;
                    }
                    ls = sycl::reduce_over_group(sg, ls, sycl::plus<float>());
                    if (lane == 0) redl[size_t(hh)*NSG + sgid] = ls;
                    if (n_idx == 0) { corrs[hh] = cr; mnews[hh] = mn; }
                }
                it.barrier(sycl::access::fence_space::local_space);
                if (tid < HG) {
                    float l_tile = 0.f;
                    const int mi = tid / TM;
#pragma unroll
                    for (int c = 0; c < NSGC; ++c) l_tile += redl[size_t(tid)*NSG + mi*NSGC + c];
                    lst[tid] = lst[tid]*corrs[tid] + l_tile;
                    mst[tid] = mnews[tid];
                }
                const int nend = sycl::min(TW, n_kv - p0);
                for (int d0 = n_idx*TD; d0 < kvl; d0 += NLC*TD) {
                    float a[TM][TD];
#pragma unroll
                    for (int i = 0; i < TM; ++i) {
                        const float cr = corrs[hb + i];
#pragma unroll
                        for (int e = 0; e < TD; ++e)
                            a[i][e] = accs[size_t(hb + i)*kvl + d0 + e]*cr;
                    }
                    for (int j = 0; j < nend; ++j) {
                        float cd[TD];
#pragma unroll
                        for (int e = 0; e < TD; ++e) cd[e] = ckv[int64_t(p0 + j)*kvl + d0 + e];
#pragma unroll
                        for (int i = 0; i < TM; ++i) {
                            const float w = wls[size_t(hb + i)*TW + j];
#pragma unroll
                            for (int e = 0; e < TD; ++e) a[i][e] += w*cd[e];
                        }
                    }
#pragma unroll
                    for (int i = 0; i < TM; ++i)
#pragma unroll
                        for (int e = 0; e < TD; ++e)
                            accs[size_t(hb + i)*kvl + d0 + e] = a[i][e];
                }
                it.barrier(sycl::access::fence_space::local_space);
            }
            for (int idx = tid; idx < HG*kvl; idx += WG) {
                const int hh = idx / kvl, d = idx - hh*kvl;
                oc[(int64_t(t)*nh + head0 + hh)*kvl + d] = accs[idx]/lst[hh];
            }
        };
        h.parallel_for(sycl::nd_range<1>(size_t(T)*ngh*WG, WG),
                       sycl::ext::oneapi::experimental::properties{
                           sycl::ext::intel::experimental::grf_size<256>}, body);
    });
}

template <int HG, int WG, int TP = 1>
static sycl::event k_mla_flash_TH(sycl::queue& q, const float* qc, const float* qr,
                                  const float* ckv, const float* kr, float* oc,
                                  int nh, int kvl, int rl, float scale,
                                  int T, int pos0) {
    const int ngh = nh / HG;
    const int qd  = kvl + rl;
    constexpr int TW = WG*TP;              // KV positions per tile
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float,1> qs  (sycl::range<1>(size_t(HG)*qd), h);
        sycl::local_accessor<float,1> accs(sycl::range<1>(size_t(HG)*kvl), h);
        sycl::local_accessor<float,1> wls (sycl::range<1>(size_t(HG)*TW), h);
        sycl::local_accessor<float,1> redm(sycl::range<1>(size_t(HG)*(WG/32)), h);
        sycl::local_accessor<float,1> redl(sycl::range<1>(size_t(HG)*(WG/32)), h);
        sycl::local_accessor<float,1> mst (sycl::range<1>(HG), h);
        sycl::local_accessor<float,1> lst (sycl::range<1>(HG), h);
        auto body = [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int64_t wg = it.get_group(0);
            const int t = int(wg / ngh);
            const int head0 = int(wg % ngh)*HG;
            const int tid = int(it.get_local_id(0));
            const int n_kv = pos0 + t + 1;
            auto sg = it.get_sub_group();
            const int sgid = int(sg.get_group_id()[0]), lane = int(sg.get_local_id()[0]);

            for (int idx = tid; idx < HG*qd; idx += WG) {
                const int hh = idx / qd, jj = idx - hh*qd;
                qs[idx] = (jj < kvl) ? qc[(int64_t(t)*nh + head0 + hh)*kvl + jj]
                                     : qr[(int64_t(t)*nh + head0 + hh)*rl + (jj - kvl)];
            }
            for (int idx = tid; idx < HG*kvl; idx += WG) accs[idx] = 0.f;
            if (tid < HG) { mst[tid] = -INFINITY; lst[tid] = 0.f; }
            it.barrier(sycl::access::fence_space::local_space);

            for (int p0 = 0; p0 < n_kv; p0 += TW) {
                // Each lane owns TP adjacent KV positions, so a q value read from
                // SLM feeds TP MACs and a KV element feeds HG: (HG+TP) loads per
                // HG*TP MACs instead of ~1 per MAC.
                const int pb = p0 + tid*TP;
                float v[HG][TP];
#pragma unroll
                for (int hh = 0; hh < HG; ++hh)
#pragma unroll
                    for (int k = 0; k < TP; ++k) v[hh][k] = 0.f;
                for (int j = 0; j < kvl; ++j) {
                    float cj[TP];
#pragma unroll
                    for (int k = 0; k < TP; ++k)
                        cj[k] = (pb + k < n_kv) ? ckv[int64_t(pb + k)*kvl + j] : 0.f;
#pragma unroll
                    for (int hh = 0; hh < HG; ++hh) {
                        const float qv = qs[size_t(hh)*qd + j];
#pragma unroll
                        for (int k = 0; k < TP; ++k) v[hh][k] += qv*cj[k];
                    }
                }
                for (int j = 0; j < rl; ++j) {
                    float rj[TP];
#pragma unroll
                    for (int k = 0; k < TP; ++k)
                        rj[k] = (pb + k < n_kv) ? kr[int64_t(pb + k)*rl + j] : 0.f;
#pragma unroll
                    for (int hh = 0; hh < HG; ++hh) {
                        const float qv = qs[size_t(hh)*qd + kvl + j];
#pragma unroll
                        for (int k = 0; k < TP; ++k) v[hh][k] += qv*rj[k];
                    }
                }
#pragma unroll
                for (int hh = 0; hh < HG; ++hh)
#pragma unroll
                    for (int k = 0; k < TP; ++k)
                        v[hh][k] = (pb + k < n_kv) ? v[hh][k]*scale : -INFINITY;

#pragma unroll
                for (int hh = 0; hh < HG; ++hh) {
                    float lm = -INFINITY;
#pragma unroll
                    for (int k = 0; k < TP; ++k) lm = sycl::max(lm, v[hh][k]);
                    lm = sycl::reduce_over_group(sg, lm, sycl::maximum<float>());
                    if (lane == 0) redm[size_t(hh)*(WG/32) + sgid] = lm;
                }
                it.barrier(sycl::access::fence_space::local_space);
                float m_new[HG], corr[HG];
#pragma unroll
                for (int hh = 0; hh < HG; ++hh) {
                    float m_tile = -INFINITY;
                    for (int i = 0; i < WG/32; ++i)
                        m_tile = sycl::max(m_tile, redm[size_t(hh)*(WG/32) + i]);
                    const float m_old = mst[hh];
                    m_new[hh] = sycl::max(m_old, m_tile);
                    corr[hh]  = (m_old == -INFINITY) ? 0.f : sycl::exp(m_old - m_new[hh]);
                    float ls = 0.f;
#pragma unroll
                    for (int k = 0; k < TP; ++k) {
                        const float w = (pb + k < n_kv) ? sycl::exp(v[hh][k] - m_new[hh]) : 0.f;
                        wls[size_t(hh)*TW + tid*TP + k] = w;
                        ls += w;
                    }
                    ls = sycl::reduce_over_group(sg, ls, sycl::plus<float>());
                    if (lane == 0) redl[size_t(hh)*(WG/32) + sgid] = ls;
                }
                it.barrier(sycl::access::fence_space::local_space);
                if (tid == 0)
#pragma unroll
                    for (int hh = 0; hh < HG; ++hh) {
                        float l_tile = 0.f;
                        for (int i = 0; i < WG/32; ++i) l_tile += redl[size_t(hh)*(WG/32) + i];
                        lst[hh] = lst[hh]*corr[hh] + l_tile;
                        mst[hh] = m_new[hh];
                    }
                const int nend = sycl::min(TW, n_kv - p0);
                for (int d = tid; d < kvl; d += WG) {
                    float a[HG];
#pragma unroll
                    for (int hh = 0; hh < HG; ++hh) a[hh] = accs[size_t(hh)*kvl + d]*corr[hh];
                    for (int j = 0; j < nend; ++j) {
                        const float cd = ckv[int64_t(p0 + j)*kvl + d];
#pragma unroll
                        for (int hh = 0; hh < HG; ++hh) a[hh] += wls[size_t(hh)*TW + j]*cd;
                    }
#pragma unroll
                    for (int hh = 0; hh < HG; ++hh) accs[size_t(hh)*kvl + d] = a[hh];
                }
                it.barrier(sycl::access::fence_space::local_space);
            }
            for (int idx = tid; idx < HG*kvl; idx += WG) {
                const int hh = idx / kvl, d = idx - hh*kvl;
                oc[(int64_t(t)*nh + head0 + hh)*kvl + d] = accs[idx]/lst[hh];
            }
        };
        h.parallel_for(sycl::nd_range<1>(size_t(T)*ngh*WG, WG),
                       sycl::ext::oneapi::experimental::properties{
                           sycl::ext::intel::experimental::grf_size<256>}, body);
    });
}

static sycl::event k_mla_flash_T(sycl::queue& q, const float* qc, const float* qr,
                                 const float* ckv, const float* kr, float* oc,
                                 int nh, int kvl, int rl, float scale,
                                 int T, int pos0) {
    constexpr int WG = 128;                 // also the KV tile width
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float,1> acc(sycl::range<1>(size_t(kvl)), h);
        sycl::local_accessor<float,1> wl (sycl::range<1>(WG), h);
        sycl::local_accessor<float,1> red(sycl::range<1>(WG/32), h);
        sycl::local_accessor<float,1> st (sycl::range<1>(2), h);
        h.parallel_for(sycl::nd_range<1>(size_t(T)*nh*WG, WG), [=](sycl::nd_item<1> it)
                       [[sycl::reqd_sub_group_size(32)]] {
            const int64_t wg = it.get_group(0);
            const int t = int(wg / nh);
            const int head = int(wg % nh);
            const int tid = int(it.get_local_id(0));
            const int n_kv = pos0 + t + 1;
            const float* qch = qc + (int64_t(t)*nh + head)*kvl;
            const float* qrh = qr + (int64_t(t)*nh + head)*rl;
            auto sg = it.get_sub_group();

            for (int j = tid; j < kvl; j += WG) acc[j] = 0.f;
            if (tid == 0) { st[0] = -INFINITY; st[1] = 0.f; }
            it.barrier(sycl::access::fence_space::local_space);

            // One KV position per thread per tile: the scores of a whole tile are
            // computed in parallel and folded into the running softmax with a
            // handful of barriers, instead of one barrier pair per position.
            for (int p0 = 0; p0 < n_kv; p0 += WG) {
                const int p = p0 + tid;
                float sc2 = -INFINITY;
                if (p < n_kv) {
                    const float* c = ckv + int64_t(p)*kvl;
                    const float* r = kr  + int64_t(p)*rl;
                    float v = 0.f;
                    for (int j = 0; j < kvl; ++j) v += qch[j]*c[j];
                    for (int j = 0; j < rl;  ++j) v += qrh[j]*r[j];
                    sc2 = v*scale;
                }
                float mx = sycl::reduce_over_group(sg, sc2, sycl::maximum<float>());
                if (sg.get_local_id()[0]==0) red[sg.get_group_id()[0]] = mx;
                it.barrier(sycl::access::fence_space::local_space);
                float m_tile = -INFINITY;
                for (int i = 0; i < WG/32; ++i) m_tile = sycl::max(m_tile, red[i]);
                // RACE FIX: `red` is reused below for the sum reduction. Without
                // this barrier a sub-group that finishes early overwrites red[]
                // while another is still reading it for m_tile, so the running
                // softmax max is wrong for that tile. Measured: the kernel
                // disagreed with ITSELF run to run on ~0.6% of elements, by up to
                // 1.07 against an output RMS of 0.013.
                it.barrier(sycl::access::fence_space::local_space);
                const float m_old = st[0];
                const float m_new = sycl::max(m_old, m_tile);
                const float corr  = (m_old == -INFINITY) ? 0.f : sycl::exp(m_old - m_new);
                wl[tid] = (p < n_kv) ? sycl::exp(sc2 - m_new) : 0.f;
                float ls = sycl::reduce_over_group(sg, wl[tid], sycl::plus<float>());
                if (sg.get_local_id()[0]==0) red[sg.get_group_id()[0]] = ls;
                it.barrier(sycl::access::fence_space::local_space);
                float l_tile = 0.f;
                for (int i = 0; i < WG/32; ++i) l_tile += red[i];
                // each thread owns a slice of the 512-wide accumulator
                for (int d = tid; d < kvl; d += WG) {
                    float a2 = acc[d]*corr;
                    const int nend = sycl::min(WG, n_kv - p0);
                    for (int j = 0; j < nend; ++j)
                        a2 += wl[j]*ckv[int64_t(p0 + j)*kvl + d];
                    acc[d] = a2;
                }
                if (tid == 0) { st[1] = st[1]*corr + l_tile; st[0] = m_new; }
                it.barrier(sycl::access::fence_space::local_space);
            }
            const float inv = 1.f/st[1];
            float* och = oc + (int64_t(t)*nh + head)*kvl;
            for (int j = tid; j < kvl; j += WG) och[j] = acc[j]*inv;
        });
    });
}

// ---------------------------------------------------------------- weights

struct QW {                       // a quantised matrix living in VRAM
    uint8_t* dev = nullptr;
    int64_t  K = 0, N = 0, row_bytes = 0;
    DType    dt = DType::kF32;
};

struct Layer {
    float *attn_norm=nullptr, *q_a_norm=nullptr, *kv_a_norm=nullptr, *ffn_norm=nullptr;
    QW wq_a, wq_b, wkv_a, wo, wk_b, wv_b;
    // Tensor-parallel halves on GPU1: the second card holds no attention weights
    // at all, so 13.7 s of a 28.5 s prefill (attention + projections) runs on one
    // card while the other idles. wq_b/wk_b/wv_b split by HEAD; wo splits along K
    // because card d owns heads [32d, 32d+32), i.e. exactly wo's input columns
    // [8192d, 8192d+8192).
    QW wq_b1, wk_b1, wv_b1, wo0h, wo0b, wo1h;
    QW ffn_gate, ffn_up, ffn_down;                 // dense blocks
    QW router;                                     // Q8_0
    float* exp_probs_b = nullptr;
    QW sh_gate, sh_up, sh_down;                    // shared expert
    QW shd0, shd0b, shg1, shu1, shd1;              // tensor-parallel halves
    const uint8_t *host_gate=nullptr, *host_up=nullptr, *host_down=nullptr;
    uint64_t off_gate=0, off_up=0, off_down=0;   // absolute byte offsets in the GGUF
    // Router bias is read every token on the host (top-k is a 256-element sort,
    // cheaper on the CPU than a device kernel + round-trip), so keep a host copy
    // rather than pulling it back from VRAM per block per token.
    std::vector<float> h_bias;
};

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) { std::fprintf(stderr, "usage: %s <gguf> [n_pred] [cache]\n", argv[0]); return 2; }
    const char* path = argv[1];
    const int n_pred = (argc > 2) ? atoi(argv[2]) : 16;
    const int cache_experts = (argc > 3) ? atoi(argv[3]) : 1200;

    g_probe = getenv("IE_GLM52_PROBE") != nullptr;
    // Read before the tier is allocated: it selects the tier's memory kind.
    if (const char* e0 = getenv("IE_GLM52_STREAM")) g_stream = atoi(e0);
    // IE_GLM52_PTRACE=1: allow any same-user process to attach (gdb stack
    // sampling under yama ptrace_scope=1).
    if (getenv("IE_GLM52_PTRACE")) prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);

    std::vector<int> prompt = {7984,264,13020,729,429,4675,279,308,7563,79043,1372,13};
    for (int i = 4; i < argc; ++i) prompt.push_back(atoi(argv[i]));

    GgufReader g;
    if (auto e = g.open(path); !e.empty()) { std::fprintf(stderr,"open: %s\n", e.c_str()); return 1; }
    // IE_GLM52_PROMPT_FILE=<text file>: tokenize it with the model's own BPE and
    // use that as the prompt (truncated to IE_GLM52_PROMPT_TOKENS if set).
    // Replaces the 2000-integer argv lists the long-prompt benchmarks needed.
    if (const char* pfile = getenv("IE_GLM52_PROMPT_FILE")) {
        std::string ptext;
        if (FILE* pfh = fopen(pfile, "rb")) {
            char pbuf[65536]; size_t pn;
            while ((pn = fread(pbuf, 1, sizeof pbuf, pfh)) > 0) ptext.append(pbuf, pn);
            fclose(pfh);
        } else { std::fprintf(stderr, "prompt file open failed: %s\n", pfile); return 1; }
        Tokenizer ptok;
        if (auto e = ptok.load_from_gguf(g); !e.empty()) {
            std::fprintf(stderr, "tokenizer: %s\n", e.c_str()); return 1;
        }
        const std::vector<int32_t> pids = ptok.encode(ptext, /*allow_special=*/false);
        prompt.assign(pids.begin(), pids.end());
        if (const char* ntok = getenv("IE_GLM52_PROMPT_TOKENS")) {
            const size_t pcap = size_t(atoll(ntok));
            if (prompt.size() > pcap) prompt.resize(pcap);
        }
        std::printf("prompt file: %s -> %zu tokens\n", pfile, prompt.size());
    }
    GlmDsaConfig C;
    if (auto e = read_glmdsa_config(g, C); !e.empty()) { std::fprintf(stderr,"cfg: %s\n", e.c_str()); return 1; }

    const int NL   = int(C.n_transformer_layers());       // 78 (MTP block skipped)
    const int H    = int(C.hidden);                        // 6144
    const int NH   = int(C.n_q_heads);                     // 64
    const int KVL  = int(C.kv_lora_rank);                  // 512
    const int RL   = int(C.rope_dim);                      // 64
    const int NOPE = int(C.key_len_mla) - RL;              // 192
    const int VHD  = int(C.value_len_mla);                 // 256
    const int QL   = int(C.q_lora_rank);                   // 2048
    const int EF   = int(C.expert_ffn);                    // 2048
    const int NE   = int(C.n_experts);                     // 256
    const int TOPK = int(C.n_experts_used);                // 8
    const int VOC  = int(C.vocab);
    const float kq_scale = 1.0f/std::sqrt(float(C.key_len_mla));
    const int n_ctx = int(prompt.size()) + n_pred + 8;

    std::printf("GLM-5.2  %d transformer blocks (+1 MTP skipped), hidden %d, "
                "%d heads, kv_lora %d, top-%d of %d experts\n",
                NL, H, NH, KVL, TOPK, NE);
    std::printf("kq_scale = 1/sqrt(%u) = %.6f, rope NORM theta %.0f, ctx %d\n",
                C.key_len_mla, kq_scale, C.rope_theta, n_ctx);
    if (n_ctx > int(C.indexer_top_k))
        std::printf("WARNING: ctx %d exceeds indexer top_k %u — dense MLA is no "
                    "longer exactly equivalent to DSA\n", n_ctx, C.indexer_top_k);

    std::vector<sycl::device> devs;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_info<sycl::info::platform::name>().find("Level-Zero")==std::string::npos) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("B70")!=std::string::npos) devs.push_back(d);
    }
    if (devs.empty()) { std::fprintf(stderr,"no B70\n"); return 1; }
    // IE_GLM52_GPUS=2 turns on the second card. It holds NO resident weights —
    // only an expert cache — so its whole 31.89 GiB is cache, and its PCIe link
    // (idle in the 1-GPU path) fetches half the experts. Both cards allocate from
    // ONE shared context so a single pinned host arena serves both.
    const int NG = (getenv("IE_GLM52_GPUS") && atoi(getenv("IE_GLM52_GPUS")) >= 2
                    && devs.size() >= 2) ? 2 : 1;
    devs.resize(NG);
    sycl::device dev = devs[0];
    // ONE CONTEXT PER DEVICE. A single context spanning both cards collapsed
    // host->device transfers from 10.8 GB/s to 0.17 GB/s -- expert fetch went
    // from 321 ms/token to 20,070 ms/token, a 62x regression, on the same binary
    // and the same data. Cross-device copies therefore go through a small host
    // bounce buffer (24 KiB per MoE block), which is cheap.
    std::vector<sycl::context> ctxs;
    for (int d = 0; d < NG; ++d) ctxs.emplace_back(devs[d]);
    sycl::context ctx = ctxs[0];
    sycl::queue q (ctx, devs[0], qprops());
    sycl::queue qt(ctx, devs[0], qprops());
    for (int i = 0; i < NG; ++i)
        std::printf("device %d: %s  (%.2f GiB)\n", i,
                    devs[i].get_info<sycl::info::device::name>().c_str(),
                    double(devs[i].get_info<sycl::info::device::global_mem_size>())/(1u<<30));
    std::printf("\n");

    // ---- IE_GLM52_SELFTEST: A/B the GEMM kernels without loading 196 GiB ----
    // The model lives on a spinning disk, so a full load costs ~30 minutes. This
    // runs both variants on synthetic rows of the real shapes, checks they agree,
    // and reports achieved GFLOP/s -- which is the number the blocking change is
    // trying to move.
    if (getenv("IE_GLM52_SELFTEST")) {
        {   // Can kernels dereference the expert tier? The tier is plain
            // anonymous memory + prepare_for_device_copy, which is defined for
            // COPIES. If system USM works, a decode miss can be served by
            // reading host memory inside the GEMV — no copy, no dependency, no
            // stream boundary. If not, the tier must become malloc_host USM.
            std::printf("USM aspects: host=%d shared=%d system=%d\n",
                        int(dev.has(sycl::aspect::usm_host_allocations)),
                        int(dev.has(sycl::aspect::usm_shared_allocations)),
                        int(dev.has(sycl::aspect::usm_system_allocations)));
            const size_t nb2 = 64u << 20;
            void* an2 = nullptr;
            if (posix_memalign(&an2, 2u<<20, nb2) == 0) {
                std::memset(an2, 0xAB, nb2);
                sycl::ext::oneapi::experimental::prepare_for_device_copy(an2, nb2, ctx);
                uint32_t* okd = sycl::malloc_device<uint32_t>(1, dev, ctx);
                q.memset(okd, 0, 4).wait();
                const uint8_t* srcp = static_cast<const uint8_t*>(an2);
                bool crashed = false;
                try {
                    q.parallel_for(sycl::range<1>(1024), [=](sycl::id<1> i) {
                        if (srcp[i[0]*61] == 0xAB) {
                            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device> a(okd[0]);
                            a.fetch_add(1u);
                        }
                    }).wait();
                } catch (...) { crashed = true; }
                uint32_t okh = 0;
                if (!crashed) q.memcpy(&okh, okd, 4).wait();
                std::printf("  kernel read of registered anonymous memory: %s (%u/1024 correct)\n",
                            crashed ? "THREW" : (okh == 1024 ? "WORKS" : "WRONG DATA"), okh);
                if (!crashed && okh == 1024) {
                    const size_t nvec = nb2/16;
                    float* acc2 = sycl::malloc_device<float>(4, dev, ctx);
                    q.memset(acc2, 0, 16).wait();
                    auto str2 = [&]{ return q.parallel_for(sycl::nd_range<1>(((nvec+255)/256)*256, 256),
                                       [=](sycl::nd_item<1> it){
                            const size_t i = it.get_global_id(0);
                            if (i >= nvec) return;
                            const sycl::vec<uint32_t,4>* v =
                                reinterpret_cast<const sycl::vec<uint32_t,4>*>(srcp);
                            const sycl::vec<uint32_t,4> x = v[i];
                            if ((x[0]|x[1]|x[2]|x[3]) == 0u) acc2[0] += 1.f; }); };
                    str2().wait();
                    auto t3 = Clock::now();
                    for (int r = 0; r < 5; ++r) str2().wait();
                    const double t3s = dsec(t3)/5;
                    std::printf("  streaming it from a kernel: %.2f ms (%.1f GB/s)\n",
                                t3s*1e3, double(nb2)/t3s/1e9);
                    sycl::free(acc2, ctx);
                }
                sycl::free(okd, ctx);
                sycl::ext::oneapi::experimental::release_from_device_copy(an2, ctx);
                std::free(an2);
            }
            // The decisive one: malloc_host USM from context 0, read by a kernel
            // on device 0 AND by a kernel on device 1 in its OWN context. The
            // engine keeps one context per card (a shared context cost 62x on
            // copies), so cross-context kernel access decides whether BOTH cards
            // can stream misses or only gpu0.
            if (devs.size() > 1) {
                const size_t nb3 = 4ull << 30;
                uint8_t* hu = sycl::malloc_host<uint8_t>(nb3, ctx);
                if (!hu) std::printf("  malloc_host(4 GiB) FAILED\n");
                else {
                    std::memset(hu, 0xC7, nb3);
                    auto check = [&](const char* tag, sycl::queue& qq, sycl::context& cc) {
                        uint32_t* ok = sycl::malloc_device<uint32_t>(1, qq.get_device(), cc);
                        qq.memset(ok, 0, 4).wait();
                        const uint8_t* sp = hu;
                        bool threw = false;
                        try {
                            qq.parallel_for(sycl::range<1>(4096), [=](sycl::id<1> i) {
                                if (sp[i[0]*997] == 0xC7) {
                                    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device> a(ok[0]);
                                    a.fetch_add(1u);
                                }
                            }).wait();
                        } catch (...) { threw = true; }
                        uint32_t okh = 0;
                        if (!threw) qq.memcpy(&okh, ok, 4).wait();
                        // bandwidth over 1 GiB
                        double gbs = 0;
                        if (!threw && okh == 4096) {
                            const size_t nv = (1ull<<30)/16;
                            float* a3 = sycl::malloc_device<float>(4, qq.get_device(), cc);
                            qq.memset(a3, 0, 16).wait();
                            auto st = [&]{ return qq.parallel_for(
                                sycl::nd_range<1>(((nv+255)/256)*256, 256), [=](sycl::nd_item<1> it){
                                    const size_t i = it.get_global_id(0);
                                    if (i >= nv) return;
                                    const sycl::vec<uint32_t,4>* v =
                                        reinterpret_cast<const sycl::vec<uint32_t,4>*>(sp);
                                    const sycl::vec<uint32_t,4> x = v[i];
                                    if ((x[0]|x[1]|x[2]|x[3]) == 0u) a3[0] += 1.f; }); };
                            st().wait();
                            auto t4 = Clock::now();
                            for (int r = 0; r < 3; ++r) st().wait();
                            gbs = double(1ull<<30)/(dsec(t4)/3)/1e9;
                            sycl::free(a3, cc);
                        }
                        std::printf("  malloc_host(ctx0) read by %s: %s (%u/4096)%s\n", tag,
                                    threw ? "THREW" : (okh == 4096 ? "WORKS" : "WRONG DATA"), okh,
                                    gbs > 0 ? (" @ " + std::to_string(gbs).substr(0,5) + " GB/s").c_str() : "");
                        sycl::free(ok, cc);
                    };
                    sycl::queue q1t(ctxs[1], devs[1], sycl::property::queue::in_order{});
                    check("device0/ctx0", q, ctxs[0]);
                    check("device1/ctx1", q1t, ctxs[1]);
                    sycl::free(hu, ctx);
                }
            }
        }
        std::printf("device 0 local mem: %zu B, max wg: %zu\n",
                    size_t(dev.get_info<sycl::info::device::local_mem_size>()),
                    size_t(dev.get_info<sycl::info::device::max_work_group_size>()));
        {   // Launch/dependency overhead. Decode issues ~1,600 launches per
            // token and its PROF phases exceed pure kernel time ~3x; this
            // separates the suspects: bare in-order launches, cross-queue deps
            // on complete events, and deps on in-flight copies.
            sycl::queue qa(ctx, dev, qprops()), qb(ctx, dev, qprops());
            float* z = sycl::malloc_device<float>(64, dev, ctx);
            q.memset(z, 0, 64*4).wait();
            const int NLNCH = 2000;
            auto tiny = [&](sycl::queue& qq, const std::vector<sycl::event>& dep) {
                return qq.submit([&](sycl::handler& h){ h.depends_on(dep);
                    h.parallel_for(sycl::range<1>(64), [=](sycl::id<1> i){ z[i[0]] += 1.f; }); });
            };
            tiny(qa, {}).wait();
            auto t0 = Clock::now();
            for (int i = 0; i < NLNCH; ++i) tiny(qa, {});
            const double sub_plain = dsec(t0)/NLNCH*1e6;
            qa.wait();
            const double plain = dsec(t0)/NLNCH*1e6;
            auto eb = tiny(qb, {}); eb.wait();
            t0 = Clock::now();
            for (int i = 0; i < NLNCH; ++i) tiny(qa, {eb});
            const double sub_x = dsec(t0)/NLNCH*1e6;
            qa.wait();
            const double xdep = dsec(t0)/NLNCH*1e6;
            const size_t nbc = 8u<<20;
            void* hsrc = sycl::malloc_host(nbc, ctx);
            uint8_t* ddst = sycl::malloc_device<uint8_t>(nbc, dev, ctx);
            std::memset(hsrc, 0, nbc);
            t0 = Clock::now();
            for (int i = 0; i < 200; ++i) {
                auto ec = qb.memcpy(ddst, hsrc, nbc);
                tiny(qa, {ec});
            }
            const double sub_live = dsec(t0)/200*1e6;
            qa.wait(); qb.wait();
            const double live = dsec(t0)/200*1e6;
            std::printf("launch overhead (us/launch, submit|complete):\n"
                        "  plain in-order        %6.1f | %6.1f\n"
                        "  dep on complete event %6.1f | %6.1f\n"
                        "  dep on in-flight 8MB copy %6.1f | %6.1f\n",
                        sub_plain, plain, sub_x, xdep, sub_live, live);
            // decode-shaped: per 'block' one 1KB D2H + host wait + 4 dep'd kernels
            std::vector<float> hb2(256);
            t0 = Clock::now();
            for (int b = 0; b < 78; ++b) {
                qa.memcpy(hb2.data(), z, 256).wait();
                auto ec = qb.memcpy(ddst, hsrc, nbc/2);
                tiny(qa, {ec}); tiny(qa, {}); tiny(qa, {}); tiny(qa, {});
            }
            qa.wait(); qb.wait();
            std::printf("  decode-shaped 78-block chain: %.1f ms/token\n", dsec(t0)*1e3);
            {   // What does a bare sycl::event cost? The decode residency loop
                // declares two per expert iteration (1,248 per token) and its
                // 44 ms/token is not copies (2 ms), not the victim search
                // (0.15 ms), not disk (the tier serves 100% of misses) and not
                // event ASSIGNMENTS (removing them changed nothing). Construction
                // and destruction are what is left.
                const int N = 20000;
                auto t0 = Clock::now();
                for (int i = 0; i < N; ++i) { sycl::event a, b; (void)a; (void)b; }
                const double ctor = dsec(t0)/N*1e6;
                sycl::event src_ev = qa.single_task([=]{}); src_ev.wait();
                sycl::event dst;
                t0 = Clock::now();
                for (int i = 0; i < N; ++i) dst = src_ev;
                const double asg = dsec(t0)/N*1e6;
                std::vector<sycl::event> v;
                t0 = Clock::now();
                for (int i = 0; i < N; ++i) { v.push_back(src_ev); if (v.size() > 64) v.clear(); }
                const double psh = dsec(t0)/N*1e6;
                std::printf("sycl::event cost: default ctor+dtor (x2) %6.3f us | assign %6.3f us "
                            "| push_back %6.3f us\n", ctor, asg, psh);
                std::printf("  -> per decode token: 624 experts x2 ctor = %.1f ms, "
                            "286 misses x2 assign = %.1f ms\n",
                            ctor*624/1000.0, asg*572/1000.0);
            }
            {   // Can a KERNEL stream host memory at link rate? If so, a miss
                // needs no copy at all: the expert GEMV reads the weights over
                // PCIe as it computes, which moves the same bytes but deletes
                // the copy/compute stream boundary that run11 showed is the
                // per-block tax — and deletes the slot write, the dependency,
                // and the host wait with it.
                const size_t hb2 = 2ull << 30;
                uint8_t* hsrc2 = sycl::malloc_host<uint8_t>(hb2, ctx);
                uint8_t* dsrc2 = sycl::malloc_device<uint8_t>(hb2, dev, ctx);
                if (hsrc2 && dsrc2) {
                    std::memset(hsrc2, 1, hb2);
                    float* acc = sycl::malloc_device<float>(64, dev, ctx);
                    qa.memset(acc, 0, 64*4).wait();
                    // stream `bytes` with a coalesced uint4 read per lane
                    auto stream = [&](const uint8_t* srcp, size_t bytes) {
                        const size_t nvec = bytes / 16;
                        return qa.submit([&](sycl::handler& h) {
                            h.parallel_for(sycl::nd_range<1>(((nvec + 255)/256)*256, 256),
                                           [=](sycl::nd_item<1> it) {
                                const size_t i = it.get_global_id(0);
                                if (i >= nvec) return;
                                const sycl::vec<uint32_t,4>* v =
                                    reinterpret_cast<const sycl::vec<uint32_t,4>*>(srcp);
                                const sycl::vec<uint32_t,4> x = v[i];
                                if ((x[0]|x[1]|x[2]|x[3]) == 0xFFFFFFFFu) acc[0] += 1.f;
                            });
                        });
                    };
                    for (size_t mb : {size_t(10), size_t(64), size_t(512)}) {
                        const size_t bytes = mb << 20;
                        stream(hsrc2, bytes).wait();
                        auto t1 = Clock::now();
                        for (int r = 0; r < 5; ++r) stream(hsrc2, bytes).wait();
                        const double th = dsec(t1)/5;
                        stream(dsrc2, bytes).wait();
                        auto t2 = Clock::now();
                        for (int r = 0; r < 5; ++r) stream(dsrc2, bytes).wait();
                        const double td = dsec(t2)/5;
                        std::printf("  kernel streams %4zu MB: host USM %7.2f ms (%5.1f GB/s)"
                                    " | device %7.2f ms (%6.1f GB/s)\n",
                                    mb, th*1e3, double(bytes)/th/1e9, td*1e3, double(bytes)/td/1e9);
                    }
                    sycl::free(acc, ctx);
                }
                if (hsrc2) sycl::free(hsrc2, ctx);
                if (dsrc2) sycl::free(dsrc2, ctx);
            }
            {   // THE allocation-count test. The MoE kernels take an array of
                // device pointers and dereference it (indirect USM access); the
                // L0 adapter must make every known allocation resident per such
                // launch. The engine holds ~4,000 slot allocations, the
                // microbenchmarks above hold ~5 — which would explain 9 us there
                // against ~1 ms/block in situ. Same kernel, same bytes, only the
                // allocation COUNT differs.
                auto run_n = [&](int nalloc, size_t each) {
                    std::vector<uint8_t*> bufs(static_cast<size_t>(nalloc), nullptr);
                    for (int i = 0; i < nalloc; ++i) {
                        bufs[size_t(i)] = sycl::malloc_device<uint8_t>(each, dev, ctx);
                        if (!bufs[size_t(i)]) { bufs.resize(size_t(i)); break; }
                    }
                    if (bufs.size() < 8) { for (auto* b : bufs) sycl::free(b, ctx);
                                           std::printf("  n=%d: alloc failed\n", nalloc); return; }
                    uint8_t** pp = sycl::malloc_shared<uint8_t*>(8, dev, ctx);
                    for (int i = 0; i < 8; ++i) pp[i] = bufs[size_t(i) % bufs.size()];
                    auto ind = [&]{                       // indirect: reads via pp[]
                        return qa.submit([&](sycl::handler& h) {
                            h.parallel_for(sycl::nd_range<1>(256, 64), [=](sycl::nd_item<1> it)
                                           [[sycl::reqd_sub_group_size(32)]] {
                                const int i = int(it.get_global_id(0));
                                float acc = 0.f;
                                for (int e = 0; e < 8; ++e) acc += float(pp[e][i & 1023]);
                                if (i == 0) z[0] = acc;
                            });
                        });
                    };
                    ind().wait();
                    auto ts = Clock::now();
                    for (int i = 0; i < 500; ++i) ind();
                    const double sub = dsec(ts)/500*1e6;
                    qa.wait();
                    const double tot = dsec(ts)/500*1e6;
                    // and a plain copy submit with the same allocation count live
                    qa.memcpy(bufs[0], hsrc, std::min<size_t>(each, 1u<<20)).wait();
                    auto ts2 = Clock::now();
                    for (int i = 0; i < 200; ++i)
                        qa.memcpy(bufs[size_t(i) % bufs.size()], hsrc, std::min<size_t>(each, 1u<<20));
                    const double csub = dsec(ts2)/200*1e6;
                    qa.wait();
                    std::printf("  allocations=%5zu: indirect-kernel submit %7.2f us | complete %7.2f us"
                                " | 1MB copy submit %7.2f us\n",
                                bufs.size(), sub, tot, csub);
                    sycl::free(pp, ctx);
                    for (auto* b : bufs) sycl::free(b, ctx);
                };
                std::printf("allocation-count scaling (indirect USM access):\n");
                const size_t esz = getenv("IE_GLM52_ALLOCSZ")
                                   ? size_t(atoll(getenv("IE_GLM52_ALLOCSZ"))) : (256u<<10);
                run_n(8,    esz);
                run_n(256,  esz);
                run_n(1024, esz);
                run_n(4000, esz);
            }
            {   // Does a cross-queue dependency poison the in-order fast
                // path? The engine's decode chains show 40-60 us GPU-side gaps
                // between consecutive in-order kernels; the clean chain above
                // shows ~2 us. Difference: decode submissions carry deps on
                // copy events from other queues every block.
                auto tiny2 = [&](sycl::queue& qq, const std::vector<sycl::event>& dep) {
                    return qq.submit([&](sycl::handler& h){ h.depends_on(dep);
                        h.parallel_for(sycl::range<1>(64), [=](sycl::id<1> i){ z[i[0]] += 1.f; }); });
                };
                const int NCH = 1000;
                // pattern A: every 5th kernel depends on a fresh in-flight copy
                auto ta = Clock::now();
                for (int i = 0; i < NCH; ++i) {
                    if (i % 5 == 0) {
                        auto ec = qb.memcpy(ddst, hsrc, 1u<<20);
                        tiny2(qa, {ec});
                    } else tiny2(qa, {});
                }
                qa.wait(); qb.wait();
                const double mixed = dsec(ta)/NCH*1e6;
                // pattern B: same but the dep is routed through a barrier
                ta = Clock::now();
                for (int i = 0; i < NCH; ++i) {
                    if (i % 5 == 0) {
                        auto ec = qb.memcpy(ddst, hsrc, 1u<<20);
                        qa.ext_oneapi_submit_barrier(std::vector<sycl::event>{ec});
                        tiny2(qa, {});
                    } else tiny2(qa, {});
                }
                qa.wait(); qb.wait();
                const double barr = dsec(ta)/NCH*1e6;
                std::printf("  chain w/ cross-queue deps every 5th: %6.2f us/kernel | via barrier: %6.2f\n",
                            mixed, barr);
            }
            {   // Copy-submission cost vs imported-region SIZE. In-engine,
                // submitting a copy from the 160 GB prepare_for_device_copy
                // tier appears to cost ~150-250 us (hostprof resolve = 66
                // ms/token); the 8 MB bench buffer costs 1.1 us. Measure the
                // scaling directly, including random source offsets.
                // HAZARD: this allocation competes with a loaded engine's
                // 160 GiB expert tier. Running it alongside a benchmark pushed
                // that tier into swap and depressed every number in the run by
                // 20-25% (run12). Opt-in only, and never while a run is loaded.
                const size_t big = getenv("IE_GLM52_BIGIMPORT")
                                   ? (size_t(atoll(getenv("IE_GLM52_BIGIMPORT"))) << 30)
                                   : (2ull << 30);
                void* bigp = nullptr;
                if (posix_memalign(&bigp, 2ull<<20, big) == 0) {
                    madvise(bigp, big, MADV_HUGEPAGE);
                    std::memset(bigp, 1, big);
                    sycl::ext::oneapi::experimental::prepare_for_device_copy(bigp, big, ctx);
                    uint8_t* dd2 = sycl::malloc_device<uint8_t>(10u<<20, dev, ctx);
                    std::mt19937_64 rg(7);
                    auto meas = [&](const char* tag, bool rnd) {
                        qa.memcpy(dd2, bigp, 10u<<20).wait();
                        auto ts = Clock::now();
                        for (int i = 0; i < 64; ++i) {
                            const size_t off = rnd ? (rg() % (big - (10u<<20))) & ~4095ull : 0;
                            qa.memcpy(dd2, static_cast<uint8_t*>(bigp) + off, 10u<<20);
                        }
                        const double sub = dsec(ts)/64*1e6;
                        qa.wait();
                        const double tot = dsec(ts)/64*1e6;
                        std::printf("  10MB H2D from 48GB import (%s): submit %7.1f us | total %7.1f us\n",
                                    tag, sub, tot);
                    };
                    meas("offset 0", false);
                    meas("random offsets", true);
                    {   // the engine registers the tier with BOTH contexts —
                        // measure whether a second-context import poisons
                        // submissions from the first.
                        sycl::context ctx2(dev);
                        sycl::ext::oneapi::experimental::prepare_for_device_copy(bigp, big, ctx2);
                        meas("dual-context import", true);
                        sycl::ext::oneapi::experimental::release_from_device_copy(bigp, ctx2);
                    }
                    sycl::ext::oneapi::experimental::release_from_device_copy(bigp, ctx);
                    std::free(bigp); sycl::free(dd2, ctx);
                }
                // and the same from a big malloc_host region
                {
                    const size_t bigh = 2ull << 30;
                    uint8_t* hp2 = sycl::malloc_host<uint8_t>(bigh, ctx);
                    if (hp2) {
                        std::memset(hp2, 1, bigh);
                        uint8_t* dd3 = sycl::malloc_device<uint8_t>(10u<<20, dev, ctx);
                        std::mt19937_64 rg2(9);
                        qa.memcpy(dd3, hp2, 10u<<20).wait();
                        auto ts = Clock::now();
                        for (int i = 0; i < 64; ++i) {
                            const size_t off = (rg2() % (bigh - (10u<<20))) & ~4095ull;
                            qa.memcpy(dd3, hp2 + off, 10u<<20);
                        }
                        const double sub = dsec(ts)/64*1e6;
                        qa.wait();
                        const double tot = dsec(ts)/64*1e6;
                        std::printf("  10MB H2D from 8GB malloc_host (random): submit %7.1f us | total %7.1f us\n",
                                    sub, tot);
                        sycl::free(hp2, ctx); sycl::free(dd3, ctx);
                    }
                }
            }
            {   // Realistic-kernel submission cost: the 6 us "plain" number
                // came from a 1-arg kernel; real MLA kernels marshal ~15
                // pointers + a local accessor per submit. If THIS costs
                // 40-80 us, decode's ~0.8 ms/block host gap is submission
                // and the fix is recorded command graphs.
                float *a1=z,*a2=z,*a3=z,*a4=z,*a5=z,*a6=z,*a7=z,*a8=z,
                      *a9=z,*a10=z,*a11=z,*a12=z;
                auto fat = [&](sycl::queue& qq) {
                    return qq.submit([&](sycl::handler& h) {
                        sycl::local_accessor<float,1> slm(sycl::range<1>(1024), h);
                        h.parallel_for(sycl::nd_range<1>(256, 64), [=](sycl::nd_item<1> it)
                                       [[sycl::reqd_sub_group_size(32)]] {
                            const int i = int(it.get_global_id(0));
                            slm[i % 1024] = a1[i%64]+a2[i%64]+a3[i%64]+a4[i%64]
                                          +a5[i%64]+a6[i%64]+a7[i%64]+a8[i%64];
                            it.barrier(sycl::access::fence_space::local_space);
                            if (i == 0) z[0] = slm[0]+a9[0]+a10[0]+a11[0]+a12[0];
                        });
                    });
                };
                fat(qa).wait();
                auto tf = Clock::now();
                for (int i = 0; i < 1000; ++i) fat(qa);
                const double fsub = dsec(tf)/1000*1e6;
                qa.wait();
                const double ftot = dsec(tf)/1000*1e6;
                std::printf("  fat kernel (12 ptrs + SLM): submit %6.1f us | complete %6.1f us\n",
                            fsub, ftot);
            }
            {   // memcpy SUBMISSION cost by source type. Decode shows ~0.6-0.9 ms
                // per block between the last copy issue and the first kernel;
                // if submitting a copy from prepare_for_device_copy-imported
                // anonymous memory costs ~100 us on the host, that is the gap.
                uint8_t* dd = sycl::malloc_device<uint8_t>(nbc, dev, ctx);
                auto probe_src = [&](const char* tag, void* srcp) {
                    qa.memcpy(dd, srcp, nbc).wait();
                    auto ts = Clock::now();
                    for (int i = 0; i < 64; ++i) qa.memcpy(dd, srcp, nbc);
                    const double sub = dsec(ts)/64*1e6;
                    qa.wait();
                    const double tot = dsec(ts)/64*1e6;
                    std::printf("  8MB H2D from %-24s submit %7.1f us | total %7.1f us (%.1f GB/s)\n",
                                tag, sub, tot, double(nbc)/(tot*1e-6)/1e9);
                };
                probe_src("malloc_host", hsrc);
                void* an = nullptr;
                if (posix_memalign(&an, 2u<<20, nbc) == 0) {
                    std::memset(an, 1, nbc);
                    probe_src("plain anonymous", an);
                    sycl::ext::oneapi::experimental::prepare_for_device_copy(an, nbc, ctx);
                    probe_src("anonymous+prepare", an);
                    sycl::ext::oneapi::experimental::release_from_device_copy(an, ctx);
                    std::free(an);
                }
                sycl::free(dd, ctx);
            }
#ifdef SYCL_EXT_ONEAPI_GRAPH
            {   // Command-graph replay: if a recorded chain of tiny kernels
                // replays at a few us/kernel, recording decode's pointer-stable
                // attention chain per block would remove most of its launch
                // overhead.
                namespace exg = sycl::ext::oneapi::experimental;
                exg::command_graph<exg::graph_state::modifiable> gr(qa.get_context(), qa.get_device());
                gr.begin_recording(qa);
                for (int i = 0; i < 12; ++i) tiny(qa, {});
                gr.end_recording();
                auto ge = gr.finalize();
                qa.ext_oneapi_graph(ge).wait();
                auto tg = Clock::now();
                for (int r = 0; r < 500; ++r) qa.ext_oneapi_graph(ge);
                const double sub_g = dsec(tg)/500/12*1e6;
                qa.wait();
                const double tot_g = dsec(tg)/500/12*1e6;
                std::printf("  graph replay (12-kernel graph): %6.2f | %6.2f us/kernel\n",
                            sub_g, tot_g);
            }
#else
            std::printf("  (sycl_ext_oneapi_graph not available in this compiler)\n");
#endif
            sycl::free(hsrc, ctx); sycl::free(ddst, ctx); sycl::free(z, ctx);
        }
        {   // The folded byte-sum must be identical to the four-extract form for
            // every seed the format can produce, not merely close.
            auto ref = [](uint32_t seed, int j) {
                const uint32_t a = (KA_P[j] * seed) & 0x3f3f3f3fu;
                return int(a & 0xFFu) + int((a>>8)&0xFFu) + int((a>>16)&0xFFu) +
                       int((a>>24)&0xFFu) - 126;
            };
            uint64_t bad = 0;
            for (uint32_t q16 = 0; q16 <= 0xFFFFu; ++q16) {
                const uint32_t seed = q16 + 4096u;
                for (int j = 0; j < 8; ++j)
                    if (trellis_at(seed, j) != ref(seed, j)) ++bad;
            }
            std::printf("trellis fold check: %llu mismatches over %d seeds x 8 lanes\n",
                        (unsigned long long)bad, 65536);
        }
        std::mt19937 rng(12345);
        // Every real dense shape in the graph, v1 vs the dispatcher, checked for
        // agreement as well as speed -- a fast wrong GEMM is worse than nothing.
        auto bench = [&](const char* what, bool iq2, int64_t K, int64_t N, int T) {
            const int64_t row_bytes = iq2 ? (4 + (K/256)*68) : q6_soa_row_bytes(K);
            std::vector<uint8_t> hw(size_t(row_bytes)*size_t(N));
            for (auto& b : hw) b = uint8_t(rng());
            // Random bytes in the scale fields decode to inf/NaN/denormals, which
            // both poisons the diff check and distorts the timing. Write real ones.
            if (iq2) {
                const float sc = 0.01f;
                for (int64_t r = 0; r < N; ++r) std::memcpy(hw.data()+size_t(r)*row_bytes, &sc, 4);
            } else {
                const sycl::half hd = sycl::half(0.02f);
                for (int64_t r = 0; r < N; ++r)
                    for (int64_t b2 = 0; b2 < K/32; ++b2)
                        std::memcpy(hw.data()+size_t(r)*row_bytes + size_t(b2)*2, &hd, 2);
            }
            std::vector<float> hx(size_t(T)*size_t(K));
            for (auto& v : hx) v = float(int(rng()%2001)-1000)/1000.f;
            std::vector<int32_t> ht(static_cast<size_t>(T), 0);
            for (int i = 0; i < T; ++i) ht[size_t(i)] = i;
            uint8_t* dw = sycl::malloc_device<uint8_t>(hw.size(), dev, ctx);
            float* dx = sycl::malloc_device<float>(hx.size(), dev, ctx);
            int32_t* dt = sycl::malloc_device<int32_t>(size_t(T), dev, ctx);
            float* y1 = sycl::malloc_device<float>(size_t(T)*size_t(N), dev, ctx);
            float* y2 = sycl::malloc_device<float>(size_t(T)*size_t(N), dev, ctx);
            q.memcpy(dw, hw.data(), hw.size()).wait();
            q.memcpy(dx, hx.data(), hx.size()*4).wait();
            q.memcpy(dt, ht.data(), size_t(T)*4).wait();
            auto one = [&](bool nu) {
                g_gemm_v2 = nu ? 1 : 0;
                if (iq2) k_iq2kt_gemm(q, dw, dx, K, dt, T, nu?y2:y1, K, N, {});
                else     k_q6_gemm(q, dw, dx, nu?y2:y1, K, N, row_bytes, T, {});
                q.wait();
            };
            double ms[2];
            for (int v = 0; v < 2; ++v) {
                one(v); one(v);
                auto t0 = Clock::now();
                for (int r = 0; r < 3; ++r) one(v);
                ms[v] = dsec(t0)*1e3/3.0;
            }
            std::vector<float> a(size_t(T)*size_t(N)), b(a.size());
            q.memcpy(a.data(), y1, a.size()*4).wait();
            q.memcpy(b.data(), y2, b.size()*4).wait();
            double maxabs = 0, rms = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                const double d = std::fabs(double(a[i]) - double(b[i]));
                if (d > maxabs) maxabs = d;
                rms += double(a[i])*double(a[i]);
            }
            rms = std::sqrt(rms/double(a.size()));
            const double gf = 2.0*double(K)*double(N)*double(T)/1e9;
            std::printf("  %-14s K=%-6lld N=%-7lld T=%-4d | v1 %8.2f ms (%7.1f GF/s)"
                        " | new %8.2f ms (%7.1f GF/s) | %5.2fx | maxabs %.2e vs rms %.2e\n",
                        what, (long long)K, (long long)N, T,
                        ms[0], gf/ms[0]*1e3, ms[1], gf/ms[1]*1e3, ms[0]/ms[1], maxabs, rms);
            sycl::free(dw, ctx); sycl::free(dx, ctx); sycl::free(dt, ctx);
            sycl::free(y1, ctx); sycl::free(y2, ctx);
        };
        {   // Can a host->device copy overlap a kernel at all?
            //
            // Every attempt to hide expert DMA behind compute has returned
            // exactly zero: the MoE total is flat across group sizes 16-256,
            // extra transfer queues change nothing, and splitting the transfer
            // by dependency changes nothing. One explanation covers all three --
            // that memcpys are running on the COMPUTE engine, where they cannot
            // overlap kernels by construction. This measures it directly.
            const size_t nby = size_t(256) << 20;          // 256 MiB
            void* hb = sycl::malloc_host(nby, ctx);
            std::memset(hb, 0, nby);
            uint8_t* db = sycl::malloc_device<uint8_t>(nby, dev, ctx);
            float* dz = sycl::malloc_device<float>(64, dev, ctx);
            q.memset(dz, 0, 64*4).wait();
            sycl::queue qcp(ctx, dev, sycl::property::queue::in_order{});
            const size_t nwi = 1u<<23; const int ITER = 9000;
            auto burn = [&](sycl::queue& qq) {
                return qq.parallel_for(sycl::range<1>(nwi), [=](sycl::id<1> id) {
                    float a0=float(id[0])*1e-7f, b0=1.0000001f;
                    for (int i = 0; i < ITER; ++i) { a0 = a0*b0 + b0; a0 = a0*b0 - b0; }
                    if (a0 == 12345.678f) dz[0] = a0;
                });
            };
            burn(q).wait(); qcp.memcpy(db, hb, nby).wait();
            auto t0 = Clock::now(); burn(q).wait(); const double tk = dsec(t0)*1e3;
            t0 = Clock::now(); qcp.memcpy(db, hb, nby).wait(); const double tc = dsec(t0)*1e3;
            t0 = Clock::now();
            auto ek = burn(q); auto ec = qcp.memcpy(db, hb, nby);
            ek.wait(); ec.wait();
            const double tb = dsec(t0)*1e3;
            std::printf("DMA/compute overlap: kernel %.2f ms | copy %.2f ms (%.1f GB/s)"
                        " | together %.2f ms\n  -> %s (serial would be %.2f, perfect %.2f)\n",
                        tk, tc, double(nby)/tc/1e6, tb,
                        tb < (tk+tc)*0.80 ? "OVERLAPS" : "SERIALISED",
                        tk+tc, std::max(tk,tc));
            sycl::free(hb, ctx); sycl::free(db, ctx); sycl::free(dz, ctx);
        }
        {   // Does a PIPELINE form? The overlap test above has no dependency
            // between the copy and the kernel; the engine's MoE does: kernel N
            // waits on copy N, while copy N+1 is issued behind it. If that
            // pattern serialises, no amount of grouping can hide expert DMA.
            const size_t nby = size_t(64) << 20;
            void* hb = sycl::malloc_host(nby, ctx);
            std::memset(hb, 0, nby);
            uint8_t* db[4];
            for (int i = 0; i < 4; ++i) db[i] = sycl::malloc_device<uint8_t>(nby, dev, ctx);
            float* dz = sycl::malloc_device<float>(64, dev, ctx);
            q.memset(dz, 0, 64*4).wait();
            sycl::queue qx2(ctx, dev, sycl::property::queue::in_order{});
            sycl::queue qc2(ctx, dev, sycl::property::queue::in_order{});
            const size_t nwi = 1u<<23; const int ITER = 3000;
            auto burn = [&](sycl::queue& qq, const std::vector<sycl::event>& dep) {
                return qq.submit([&](sycl::handler& h2){ h2.depends_on(dep);
                    h2.parallel_for(sycl::range<1>(nwi), [=](sycl::id<1> id) {
                        float a0=float(id[0])*1e-7f, b0=1.0000001f;
                        for (int i = 0; i < ITER; ++i) { a0 = a0*b0 + b0; a0 = a0*b0 - b0; }
                        if (a0 == 12345.678f) dz[0] = a0; }); });
            };
            burn(qc2, {}).wait(); qx2.memcpy(db[0], hb, nby).wait();
            auto t0 = Clock::now(); burn(qc2, {}).wait(); const double tk = dsec(t0)*1e3;
            t0 = Clock::now(); qx2.memcpy(db[0], hb, nby).wait(); const double tc = dsec(t0)*1e3;
            t0 = Clock::now();
            std::vector<sycl::event> pend;
            for (int i = 0; i < 4; ++i) {
                auto ec = qx2.memcpy(db[i], hb, nby);
                pend.assign(1, burn(qc2, {ec}));
                (void)pend;
            }
            qc2.wait(); qx2.wait();
            const double tp = dsec(t0)*1e3;
            std::printf("pipeline (kernel N waits on copy N, copy N+1 behind it):\n"
                        "  kernel %.2f ms | copy %.2f ms | 4 stages %.2f ms\n"
                        "  -> %s  (serial 4x(k+c)=%.2f, pipelined c+4k=%.2f)\n",
                        tk, tc, tp,
                        tp < 4*(tk+tc)*0.85 ? "PIPELINES" : "SERIALISED",
                        4*(tk+tc), tc + 4*tk);
            for (int i = 0; i < 4; ++i) sycl::free(db[i], ctx);
            sycl::free(hb, ctx); sycl::free(dz, ctx);
        }
        {   // raw fp32 FMA ceiling: independent chains, everything in registers
            const size_t nwi = 1u<<22;
            float* dz = sycl::malloc_device<float>(64, dev, ctx);
            q.memset(dz, 0, 64*4).wait();
            const int ITER = 512;
            auto fma = [&]{
                q.parallel_for(sycl::range<1>(nwi), [=](sycl::id<1> id) {
                    float a0=float(id[0])*1e-7f, a1=a0+1, a2=a0+2, a3=a0+3;
                    float b0=1.0000001f, b1=1.0000002f, b2=1.0000003f, b3=1.0000004f;
                    for (int i = 0; i < ITER; ++i) {
                        a0 = a0*b0 + b1; a1 = a1*b1 + b2; a2 = a2*b2 + b3; a3 = a3*b3 + b0;
                        a0 = a0*b1 + b2; a1 = a1*b2 + b3; a2 = a2*b3 + b0; a3 = a3*b0 + b1;
                    }
                    if (a0+a1+a2+a3 == 12345.678f) dz[0] = 1.f;
                });
                q.wait();
            };
            fma();
            auto t0 = Clock::now(); fma();
            std::printf("raw fp32 FMA ceiling: %.2f TFLOP/s\n",
                        double(nwi)*double(ITER)*16.0/dsec(t0)/1e12);
            sycl::free(dz, ctx);
        }
        {   // The router and the two MLA-absorb GEMVs were just batched over the
            // token dimension. Both are on the correctness path, so check them
            // against the per-token kernels they replace.
            std::printf("\nBatched-over-tokens kernels vs their per-token originals\n");
            const int T = 512;
            {   const int64_t K = 6144, N = 256, rb = (K/32)*34;
                std::vector<uint8_t> hw(size_t(rb)*size_t(N));
                for (auto& b : hw) b = uint8_t(rng());
                { const sycl::half hd = sycl::half(0.02f);
                  for (int64_t r = 0; r < N; ++r)
                      for (int64_t b2 = 0; b2 < K/32; ++b2)
                          std::memcpy(hw.data()+size_t(r)*rb + size_t(b2)*34, &hd, 2); }
                std::vector<float> hx(size_t(T)*size_t(K));
                for (auto& v : hx) v = float(int(rng()%2001)-1000)/1000.f;
                uint8_t* dw = sycl::malloc_device<uint8_t>(hw.size(), dev, ctx);
                float* dx = sycl::malloc_device<float>(hx.size(), dev, ctx);
                float* y1 = sycl::malloc_device<float>(size_t(T)*size_t(N), dev, ctx);
                float* y2 = sycl::malloc_device<float>(size_t(T)*size_t(N), dev, ctx);
                q.memcpy(dw, hw.data(), hw.size()).wait();
                q.memcpy(dx, hx.data(), hx.size()*4).wait();
                auto per_tok = [&]{ for (int t=0;t<T;++t)
                        k_q8_gemv(q, dw, dx + int64_t(t)*K, y1 + int64_t(t)*N, K, N); q.wait(); };
                auto batched = [&]{ k_q8_gemm_v3<64,64,32,4,4>(q, dw, dx, y2, K, N, T); q.wait(); };
                per_tok(); batched();
                auto t0 = Clock::now(); per_tok(); const double m1 = dsec(t0)*1e3;
                t0 = Clock::now(); batched(); const double m2 = dsec(t0)*1e3;
                std::vector<float> a(size_t(T)*size_t(N)), b(a.size());
                q.memcpy(a.data(), y1, a.size()*4).wait();
                q.memcpy(b.data(), y2, b.size()*4).wait();
                double mx = 0, rms = 0;
                for (size_t i=0;i<a.size();++i){ const double d=std::fabs(double(a[i])-double(b[i]));
                    if (d>mx) mx=d; rms += double(a[i])*double(a[i]); }
                std::printf("  router Q8_0 K=%lld N=%lld T=%d | per-token %d x %7.2f ms -> blocked %7.2f ms"
                            " | %5.1fx | maxabs %.2e vs rms %.2e\n", (long long)K,(long long)N,T,
                            T, m1, m2, m1/m2, mx, std::sqrt(rms/double(a.size())));
                sycl::free(dw,ctx); sycl::free(dx,ctx); sycl::free(y1,ctx); sycl::free(y2,ctx);
            }
            for (int which = 0; which < 2; ++which) {
                const int NHt = 64;
                const int64_t K = which ? 512 : 192, N = which ? 256 : 512;
                const int64_t xs = which ? 512 : 256, ys = N;
                const int64_t rb = q6_soa_row_bytes(K);
                const int64_t rowsn = int64_t(NHt)*N;
                std::vector<uint8_t> hw(size_t(rb)*size_t(rowsn));
                for (auto& b : hw) b = uint8_t(rng());
                { const sycl::half hd = sycl::half(0.02f);
                  for (int64_t r = 0; r < rowsn; ++r)
                      for (int64_t b2 = 0; b2 < K/32; ++b2)
                          std::memcpy(hw.data()+size_t(r)*rb + size_t(b2)*2, &hd, 2); }
                std::vector<float> hx(size_t(T)*size_t(NHt)*size_t(xs));
                for (auto& v : hx) v = float(int(rng()%2001)-1000)/1000.f;
                uint8_t* dw = sycl::malloc_device<uint8_t>(hw.size(), dev, ctx);
                float* dx = sycl::malloc_device<float>(hx.size(), dev, ctx);
                float* y1 = sycl::malloc_device<float>(size_t(T)*size_t(NHt)*size_t(ys), dev, ctx);
                float* y2 = sycl::malloc_device<float>(size_t(T)*size_t(NHt)*size_t(ys), dev, ctx);
                q.memcpy(dw, hw.data(), hw.size()).wait();
                q.memcpy(dx, hx.data(), hx.size()*4).wait();
                auto per_tok = [&]{ for (int t=0;t<T;++t)
                        k_q6_gemv_multi_small(q, dw, dx + int64_t(t)*NHt*xs, xs,
                                              y1 + int64_t(t)*NHt*ys, ys, K, N, rb, NHt); q.wait(); };
                auto batched = [&]{ k_q6_gemv_multi_small_T(q, dw, dx, xs, int64_t(NHt)*xs,
                                              y2, ys, int64_t(NHt)*ys, K, N, rb, NHt, T); q.wait(); };
                per_tok(); batched();
                auto t0 = Clock::now(); per_tok(); const double m1 = dsec(t0)*1e3;
                t0 = Clock::now(); batched(); const double m2 = dsec(t0)*1e3;
                std::vector<float> a(size_t(T)*size_t(NHt)*size_t(ys)), b(a.size());
                q.memcpy(a.data(), y1, a.size()*4).wait();
                q.memcpy(b.data(), y2, b.size()*4).wait();
                double mx = 0, rms = 0;
                for (size_t i=0;i<a.size();++i){ const double d=std::fabs(double(a[i])-double(b[i]));
                    if (d>mx) mx=d; rms += double(a[i])*double(a[i]); }
                std::printf("  %-5s K=%-4lld N=%-4lld x64 T=%d | %d launches %7.2f ms -> 1 launch %7.2f ms"
                            " | %5.1fx | maxabs %.2e vs rms %.2e\n", which?"wv_b":"wk_b",
                            (long long)K,(long long)N,T,T,m1,m2,m1/m2,mx,std::sqrt(rms/double(a.size())));
                sycl::free(dw,ctx); sycl::free(dx,ctx); sycl::free(y1,ctx); sycl::free(y2,ctx);
            }
        }
        {   // Flash MLA is O(T^2) and is the one prefill phase whose cost is not
            // in the GEMM budget. Measure it at the chunk sizes we actually use.
            std::printf("\nFlash MLA (prefill attention), 64 heads, kv_lora 512 + rope 64\n");
            {   // Ground both kernels in a plain fp64 CPU softmax-attention at a
                // size small enough to compute directly. Neither GPU kernel is
                // the oracle here.
                const int Tc = 384, NHc = 8, KVc = 512, RLc = 64;
                std::vector<float> hq(size_t(Tc)*NHc*KVc), hr(size_t(Tc)*NHc*RLc),
                                   hk(size_t(Tc)*KVc), hkr(size_t(Tc)*RLc);
                for (auto& v : hq)  v = float(int(rng()%2001)-1000)/3000.f;
                for (auto& v : hr)  v = float(int(rng()%2001)-1000)/3000.f;
                for (auto& v : hk)  v = float(int(rng()%2001)-1000)/3000.f;
                for (auto& v : hkr) v = float(int(rng()%2001)-1000)/3000.f;
                const float sc = 1.0f/16.0f;
                std::vector<double> ref(size_t(Tc)*NHc*KVc, 0.0);
                for (int t = 0; t < Tc; ++t)
                    for (int h2 = 0; h2 < NHc; ++h2) {
                        std::vector<double> sco(size_t(t+1));
                        double mx = -1e300;
                        for (int p2 = 0; p2 <= t; ++p2) {
                            double v = 0;
                            for (int j = 0; j < KVc; ++j)
                                v += double(hq[(size_t(t)*NHc+h2)*KVc+j])*double(hk[size_t(p2)*KVc+j]);
                            for (int j = 0; j < RLc; ++j)
                                v += double(hr[(size_t(t)*NHc+h2)*RLc+j])*double(hkr[size_t(p2)*RLc+j]);
                            sco[size_t(p2)] = v*double(sc);
                            if (sco[size_t(p2)] > mx) mx = sco[size_t(p2)];
                        }
                        double sum = 0;
                        for (auto& v : sco) { v = std::exp(v - mx); sum += v; }
                        for (int p2 = 0; p2 <= t; ++p2) {
                            const double w = sco[size_t(p2)]/sum;
                            for (int j = 0; j < KVc; ++j)
                                ref[(size_t(t)*NHc+h2)*KVc+j] += w*double(hk[size_t(p2)*KVc+j]);
                        }
                    }
                float* dq = sycl::malloc_device<float>(hq.size(), dev, ctx);
                float* dr = sycl::malloc_device<float>(hr.size(), dev, ctx);
                float* dk = sycl::malloc_device<float>(hk.size(), dev, ctx);
                float* dkr= sycl::malloc_device<float>(hkr.size(), dev, ctx);
                float* dov= sycl::malloc_device<float>(hq.size(), dev, ctx);
                q.memcpy(dq, hq.data(), hq.size()*4).wait();
                q.memcpy(dr, hr.data(), hr.size()*4).wait();
                q.memcpy(dk, hk.data(), hk.size()*4).wait();
                q.memcpy(dkr, hkr.data(), hkr.size()*4).wait();
                auto check = [&](const char* nm) {
                    std::vector<float> got(hq.size());
                    q.memcpy(got.data(), dov, got.size()*4).wait();
                    double mx2 = 0, rms = 0;
                    for (size_t i = 0; i < ref.size(); ++i) {
                        const double d = std::fabs(ref[i] - double(got[i]));
                        if (d > mx2) mx2 = d;
                        rms += ref[i]*ref[i];
                    }
                    std::printf("  vs fp64 CPU attention: %-14s maxabs %.3e (rms %.3e)\n",
                                nm, mx2, std::sqrt(rms/double(ref.size())));
                };
                q.memset(dov, 0, hq.size()*4).wait();
                k_mla_flash_T(q, dq, dr, dk, dkr, dov, NHc, KVc, RLc, sc, Tc, 0); q.wait();
                check("k_mla_flash_T");
                q.memset(dov, 0, hq.size()*4).wait();
                k_mla_flash_TH<8,128>(q, dq, dr, dk, dkr, dov, NHc, KVc, RLc, sc, Tc, 0); q.wait();
                check("k_mla_flash_TH");
                sycl::free(dq,ctx); sycl::free(dr,ctx); sycl::free(dk,ctx);
                sycl::free(dkr,ctx); sycl::free(dov,ctx);
            }
            const int NHt = 64, KVLt = 512, RLt = 64;
            for (int T : {512, 1024, 2000}) {
                const int nctx = T;
                float* qc = sycl::malloc_device<float>(size_t(T)*NHt*KVLt, dev, ctx);
                float* qr = sycl::malloc_device<float>(size_t(T)*NHt*RLt, dev, ctx);
                float* kc = sycl::malloc_device<float>(size_t(nctx)*KVLt, dev, ctx);
                float* rc = sycl::malloc_device<float>(size_t(nctx)*RLt, dev, ctx);
                float* oc = sycl::malloc_device<float>(size_t(T)*NHt*KVLt, dev, ctx);
                {   std::vector<float> h(size_t(T)*NHt*KVLt);
                    for (auto& v : h) v = float(int(rng()%2001)-1000)/3000.f;
                    q.memcpy(qc, h.data(), h.size()*4).wait();
                    q.memcpy(oc, h.data(), size_t(T)*NHt*KVLt*4).wait(); }
                {   std::vector<float> h(size_t(T)*NHt*RLt);
                    for (auto& v : h) v = float(int(rng()%2001)-1000)/3000.f;
                    q.memcpy(qr, h.data(), h.size()*4).wait(); }
                {   std::vector<float> h(size_t(nctx)*KVLt);
                    for (auto& v : h) v = float(int(rng()%2001)-1000)/3000.f;
                    q.memcpy(kc, h.data(), h.size()*4).wait(); }
                {   std::vector<float> h(size_t(nctx)*RLt);
                    for (auto& v : h) v = float(int(rng()%2001)-1000)/3000.f;
                    q.memcpy(rc, h.data(), h.size()*4).wait(); }
                float* o2 = sycl::malloc_device<float>(size_t(T)*NHt*KVLt, dev, ctx);
                const double pos = double(T)*double(T+1)/2.0;
                const double gf = pos*double(NHt)*(576.0 + 512.0)*2.0/1e9;
                auto one = [&](const char* nm, auto fn, float* dst) {
                    fn(); q.wait();
                    auto t0 = Clock::now(); fn(); q.wait();
                    const double ms = dsec(t0)*1e3;
                    std::printf("  T=%-5d %-12s %9.2f ms  %7.1f GF/s  -> %6.2f s for 78 blocks",
                                T, nm, ms, gf/ms*1e3, ms*78/1e3);
                    (void)dst;
                    return ms;
                };
                const double mbase = one("per-head", [&]{ k_mla_flash_T(q, qc, qr, kc, rc, oc,
                                        NHt, KVLt, RLt, 1.0f/16.0f, T, 0); }, oc);
                std::printf("\n");
                    if (T == 2000) {   // decode-path attention at this context length
                    float* d1 = sycl::malloc_device<float>(size_t(NHt)*KVLt, dev, ctx);
                    float* d2 = sycl::malloc_device<float>(size_t(NHt)*KVLt, dev, ctx);
                    float* sx = sycl::malloc_device<float>(size_t(NHt)*nctx, dev, ctx);
                    auto dec = [&](int v) {
                        if (v) k_mla_attend_hg<8>(q, qc, qr, kc, rc, d2, NHt, nctx,
                                                  KVLt, RLt, 1.0f/16.0f, sx);
                        else   k_mla_attend(q, qc, qr, kc, rc, d1, NHt, nctx,
                                            KVLt, RLt, 1.0f/16.0f, sx);
                        q.wait(); };
                    double dm[2];
                    for (int v = 0; v < 2; ++v) { dec(v);
                        auto t1 = Clock::now();
                        for (int r2 = 0; r2 < 5; ++r2) dec(v);
                        dm[v] = dsec(t1)*1e3/5.0; }
                    std::vector<float> x1(size_t(NHt)*KVLt), x2(x1.size());
                    q.memcpy(x1.data(), d1, x1.size()*4).wait();
                    q.memcpy(x2.data(), d2, x2.size()*4).wait();
                    double mx3 = 0, rms3 = 0;
                    for (size_t i=0;i<x1.size();++i){ const double d3=std::fabs(double(x1[i])-double(x2[i]));
                        if (d3>mx3) mx3=d3; rms3 += double(x1[i])*double(x1[i]); }
                    std::printf("  DECODE ctx=%d: per-head %6.2f ms -> HG=8 %6.2f ms (%4.2fx)"
                                " | %5.1f -> %5.1f ms/token over 78 blocks | maxabs %.2e vs rms %.2e\n",
                                nctx, dm[0], dm[1], dm[0]/dm[1], dm[0]*78, dm[1]*78,
                                mx3, std::sqrt(rms3/double(x1.size())));
                    sycl::free(d1,ctx); sycl::free(d2,ctx); sycl::free(sx,ctx);
                }
            {   // Is the old kernel even deterministic? Run it twice.
                    std::vector<float> r1(size_t(T)*NHt*KVLt), r2(r1.size());
                    q.memcpy(r1.data(), oc, r1.size()*4).wait();
                    k_mla_flash_T(q, qc, qr, kc, rc, o2, NHt, KVLt, RLt, 1.0f/16.0f, T, 0);
                    q.wait();
                    q.memcpy(r2.data(), o2, r2.size()*4).wait();
                    double mx2 = 0; size_t nb2 = 0;
                    for (size_t i=0;i<r1.size();++i){ const double d=std::fabs(double(r1[i])-double(r2[i]));
                        if (d>mx2) mx2=d; if (d>1e-3) ++nb2; }
                    std::printf("    k_mla_flash_T run-to-run: maxabs %.3e over %zu/%zu elems\n",
                                mx2, nb2, r1.size());
                }
                std::vector<float> a(size_t(T)*NHt*KVLt), b(a.size());
                q.memcpy(a.data(), oc, a.size()*4).wait();
                auto cmp = [&](double ms) {
                    q.memcpy(b.data(), o2, b.size()*4).wait();
                    double mx = 0, rms = 0; size_t at = 0; size_t nbad = 0;
                    for (size_t i=0;i<a.size();++i){ const double d=std::fabs(double(a[i])-double(b[i]));
                        if (d>mx) { mx=d; at=i; }
                        if (d > 1e-3) ++nbad;
                        rms += double(a[i])*double(a[i]); }
                    const size_t bt = at/(size_t(NHt)*KVLt), bh = (at/KVLt)%NHt, bd = at%KVLt;
                    std::printf("  %5.2fx | maxabs %.2e vs rms %.2e | worst at t=%zu head=%zu d=%zu"
                                " (ref %.4f got %.4f) | %zu/%zu elems off\n",
                                mbase/ms, mx, std::sqrt(rms/double(a.size())),
                                bt, bh, bd, a[at], b[at], nbad, a.size());
                };
                cmp(one("HG=4 WG=128", [&]{ k_mla_flash_TH<4,128>(q, qc, qr, kc, rc, o2,
                                        NHt, KVLt, RLt, 1.0f/16.0f, T, 0); }, o2));
                cmp(one("HG=8 WG=128", [&]{ k_mla_flash_TH<8,128>(q, qc, qr, kc, rc, o2,
                                        NHt, KVLt, RLt, 1.0f/16.0f, T, 0); }, o2));
                cmp(one("HG=8 WG=64",  [&]{ k_mla_flash_TH<8,64>(q, qc, qr, kc, rc, o2,
                                        NHt, KVLt, RLt, 1.0f/16.0f, T, 0); }, o2));
                cmp(one("RT 8/8x1x4",  [&]{ k_mla_flash_RT<8,128,8,1,4>(q, qc, qr, kc, rc, o2,
                                        NHt, KVLt, RLt, 1.0f/16.0f, T, 0); }, o2));
                cmp(one("RG 16/W256 TD2",[&]{ k_mla_flash_RG<16,256,2>(q, qc, qr, kc, rc, o2,
                                        NHt, KVLt, RLt, 1.0f/16.0f, T, 0); }, o2));
                cmp(one("RG 8/W256 TD2", [&]{ k_mla_flash_RG<8,256,2>(q, qc, qr, kc, rc, o2,
                                        NHt, KVLt, RLt, 1.0f/16.0f, T, 0); }, o2));
                cmp(one("RG 16/W128 TD4",[&]{ k_mla_flash_RG<16,128,4>(q, qc, qr, kc, rc, o2,
                                        NHt, KVLt, RLt, 1.0f/16.0f, T, 0); }, o2));
                cmp(one("RG 8/W128 TD4", [&]{ k_mla_flash_RG<8,128,4>(q, qc, qr, kc, rc, o2,
                                        NHt, KVLt, RLt, 1.0f/16.0f, T, 0); }, o2));
                cmp(one("RG 32/W256 TD2",[&]{ k_mla_flash_RG<32,256,2>(q, qc, qr, kc, rc, o2,
                                        NHt, KVLt, RLt, 1.0f/16.0f, T, 0); }, o2));
                cmp(one("RT 8/8x1x4 W512",[&]{ k_mla_flash_RT<8,512,8,1,4>(q, qc, qr, kc, rc, o2,
                                        NHt, KVLt, RLt, 1.0f/16.0f, T, 0); }, o2));
                cmp(one("RT 8/8x1x2 W256",[&]{ k_mla_flash_RT<8,256,8,1,2>(q, qc, qr, kc, rc, o2,
                                        NHt, KVLt, RLt, 1.0f/16.0f, T, 0); }, o2));
                cmp(one("RT 8/8x1x8 W256",[&]{ k_mla_flash_RT<8,256,8,1,8>(q, qc, qr, kc, rc, o2,
                                        NHt, KVLt, RLt, 1.0f/16.0f, T, 0); }, o2));
                cmp(one("RT 8/8x2x4 W256",[&]{ k_mla_flash_RT<8,256,8,2,4>(q, qc, qr, kc, rc, o2,
                                        NHt, KVLt, RLt, 1.0f/16.0f, T, 0); }, o2));
                cmp(one("RT 8/8x1x4 W256",[&]{ k_mla_flash_RT<8,256,8,1,4>(q, qc, qr, kc, rc, o2,
                                        NHt, KVLt, RLt, 1.0f/16.0f, T, 0); }, o2));
                cmp(one("RT 8/8x1x4 W64",[&]{ k_mla_flash_RT<8,64,8,1,4>(q, qc, qr, kc, rc, o2,
                                        NHt, KVLt, RLt, 1.0f/16.0f, T, 0); }, o2));
                cmp(one("RT 16/16x1x4",[&]{ k_mla_flash_RT<16,128,16,1,4>(q, qc, qr, kc, rc, o2,
                                        NHt, KVLt, RLt, 1.0f/16.0f, T, 0); }, o2));
                cmp(one("RT 16/16x1x2",[&]{ k_mla_flash_RT<16,128,16,1,2>(q, qc, qr, kc, rc, o2,
                                        NHt, KVLt, RLt, 1.0f/16.0f, T, 0); }, o2));
                cmp(one("RT 16/8x1x4", [&]{ k_mla_flash_RT<16,128,8,1,4>(q, qc, qr, kc, rc, o2,
                                        NHt, KVLt, RLt, 1.0f/16.0f, T, 0); }, o2));
                cmp(one("RT 4/4x1x4",  [&]{ k_mla_flash_RT<4,128,4,1,4>(q, qc, qr, kc, rc, o2,
                                        NHt, KVLt, RLt, 1.0f/16.0f, T, 0); }, o2));
                sycl::free(qc,ctx); sycl::free(qr,ctx); sycl::free(kc,ctx);
                sycl::free(rc,ctx); sycl::free(oc,ctx);
            }
        }
        std::printf("\nDense GEMM: every real shape in the graph\n");
        for (int T : {64, 512, 1024}) {
            bench("wq_a",    false, 6144,  2048,   T);
            bench("wq_b",    false, 2048,  12288,  T);
            bench("wkv_a",   false, 6144,  576,    T);
            bench("wo",      false, 16384, 6144,   T);
            bench("sh_down", false, 2048,  6144,   T);
            bench("ffn_up",  false, 6144,  12288,  T);
            bench("lm_head", false, 6144,  154880, T);
        }
        {   // Q6 register-tile sweep: wv[32] + acc[TMAX] may be spilling
            std::printf("\nQ6_0 register-tile sweep (K=16384 N=6144 T=512)\n");
            const int64_t K = 16384, N = 6144; const int T = 512;
            const int64_t row_bytes = q6_soa_row_bytes(K);
            std::vector<uint8_t> hw(size_t(row_bytes)*size_t(N));
            for (auto& b : hw) b = uint8_t(rng());
            {   const sycl::half hd = sycl::half(0.02f);
                for (int64_t r = 0; r < N; ++r)
                    for (int64_t b2 = 0; b2 < K/32; ++b2)
                        std::memcpy(hw.data()+size_t(r)*row_bytes + size_t(b2)*2, &hd, 2); }
            std::vector<float> hx(size_t(T)*size_t(K));
            for (auto& v : hx) v = float(int(rng()%2001)-1000)/1000.f;
            uint8_t* dw = sycl::malloc_device<uint8_t>(hw.size(), dev, ctx);
            float* dx = sycl::malloc_device<float>(hx.size(), dev, ctx);
            float* yy = sycl::malloc_device<float>(size_t(T)*size_t(N), dev, ctx);
            q.memcpy(dw, hw.data(), hw.size()).wait();
            q.memcpy(dx, hx.data(), hx.size()*4).wait();
            const double gf = 2.0*double(K)*double(N)*double(T)/1e9;
            auto tile = [&](const char* nm, auto fn) {
                fn(); fn(); q.wait();
                auto t0 = Clock::now();
                for (int r = 0; r < 3; ++r) fn();
                q.wait();
                const double ms = dsec(t0)*1e3/3.0;
                std::printf("  %-16s %8.2f ms  %7.1f GF/s\n", nm, ms, gf/ms*1e3);
            };
            tile("TMAX=4  WG=128",  [&]{ k_q6_gemm_v1t<4,128>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("TMAX=8  WG=128",  [&]{ k_q6_gemm_v1t<8,128>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("TMAX=16 WG=128",  [&]{ k_q6_gemm_v1t<16,128>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("TMAX=8  WG=256",  [&]{ k_q6_gemm_v1t<8,256>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("TMAX=8  WG=64",   [&]{ k_q6_gemm_v1t<8,64>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("TMAX=32 WG=128",  [&]{ k_q6_gemm_v1t<32,128>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            std::printf("  -- v3 register-blocked BMxBN / TMxTN --\n");
            tile("v3 64x64  4x4 K64",  [&]{ k_q6_gemm_v3<64,64,64,4,4>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("v3 64x64  4x4 K32",  [&]{ k_q6_gemm_v3<64,64,32,4,4>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("v3 64x64  4x4 K128", [&]{ k_q6_gemm_v3<64,64,128,4,4>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("v3 64x128 4x4 K64",  [&]{ k_q6_gemm_v3<64,128,64,4,4>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("v3 128x64 4x4 K64",  [&]{ k_q6_gemm_v3<128,64,64,4,4>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("v3 128x128 4x4 K64", [&]{ k_q6_gemm_v3<128,128,64,4,4>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("v3 64x64  4x8 K64",  [&]{ k_q6_gemm_v3<64,64,64,4,8>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("v3 64x64  8x4 K64",  [&]{ k_q6_gemm_v3<64,64,64,8,4>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("v3 64x64  2x4 K64",  [&]{ k_q6_gemm_v3<64,64,64,2,4>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("v3 64x64  4x2 K64",  [&]{ k_q6_gemm_v3<64,64,64,4,2>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("v3 32x64  4x4 K64",  [&]{ k_q6_gemm_v3<32,64,64,4,4>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("v3 64x128 4x4 K32",  [&]{ k_q6_gemm_v3<64,128,32,4,4>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("v3 128x128 4x4 K32", [&]{ k_q6_gemm_v3<128,128,32,4,4>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("v3 64x256 4x4 K32",  [&]{ k_q6_gemm_v3<64,256,32,4,4>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("v3 64x256 4x4 K64",  [&]{ k_q6_gemm_v3<64,256,64,4,4>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            std::printf("  -- 256-GRF: does the larger register budget unlock an 8x8 tile? --\n");
            tile("grf 64x256 4x4",  [&]{ k_q6_gemm_v3<64,256,32,4,4,true>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("grf 64x128 8x8",  [&]{ k_q6_gemm_v3<64,128,32,8,8,true>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("grf 128x128 8x8", [&]{ k_q6_gemm_v3<128,128,32,8,8,true>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("grf 128x256 8x8", [&]{ k_q6_gemm_v3<128,256,32,8,8,true>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("grf 64x128 4x8",  [&]{ k_q6_gemm_v3<64,128,32,4,8,true>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("grf 128x128 8x4", [&]{ k_q6_gemm_v3<128,128,32,8,4,true>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            std::printf("  -- XMX/DPAS, fp16 operands, fp32 accumulate --\n");
            tile("xmx 64x128 K128", [&]{ k_q6_gemm_xmx<64,128,128>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("xmx 64x256 K128", [&]{ k_q6_gemm_xmx<64,256,128>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("xmx 32x128 K128", [&]{ k_q6_gemm_xmx<32,128,128>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("xmx 64x128 K256", [&]{ k_q6_gemm_xmx<64,128,256>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("xmx 128x128 K64", [&]{ k_q6_gemm_xmx<128,128,64>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("xmx 64x64  K128", [&]{ k_q6_gemm_xmx<64,64,128>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            std::printf("  -- fp16 SLM tiles (fp32 accumulate) --\n");
            tile("h 128x256 8x8",  [&]{ k_q6_gemm_v3<128,256,32,8,8,true,true>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("h 128x512 8x8",  [&]{ k_q6_gemm_v3<128,512,32,8,8,true,true>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("h 256x256 8x8",  [&]{ k_q6_gemm_v3<256,256,32,8,8,true,true>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("h 128x256 8x8 K64",[&]{ k_q6_gemm_v3<128,256,64,8,8,true,true>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("h 64x256 4x4",   [&]{ k_q6_gemm_v3<64,256,32,4,4,true,true>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            tile("v3 32x256 4x4 K32",  [&]{ k_q6_gemm_v3<32,256,32,4,4>(q,dw,dx,yy,K,N,row_bytes,T,{}); });
            {   // correctness of the winner shape against v1
                float* yr = sycl::malloc_device<float>(size_t(T)*size_t(N), dev, ctx);
                k_q6_gemm_v1(q,dw,dx,yr,K,N,row_bytes,T,{}); q.wait();
                k_q6_gemm_v3<64,64,64,8,8>(q,dw,dx,yy,K,N,row_bytes,T,{}); q.wait();
                std::vector<float> a(size_t(T)*size_t(N)), b(a.size());
                q.memcpy(a.data(), yr, a.size()*4).wait();
                q.memcpy(b.data(), yy, b.size()*4).wait();
                double mr = 0, rms = 0;
                for (size_t i=0;i<a.size();++i){ rms += double(a[i])*double(a[i]);
                    const double dd = std::fabs(double(a[i])-double(b[i]));
                    if (dd > mr) mr = dd; }
                rms = std::sqrt(rms/double(a.size()));
                std::printf("  v3 vs v1: max abs diff %.3e (output rms %.3e)\n", mr, rms);
                sycl::free(yr, ctx);
            }
            sycl::free(dw, ctx); sycl::free(dx, ctx); sycl::free(yy, ctx);
        }
        {   // IQ2_KT expert GEMM: same tiling, at the token counts prefill really
            // gives one expert (about T/32 of the chunk).
            std::printf("\nIQ2_KT expert-GEMM sweep (v1 vs register-blocked v3)\n");
            for (int shape = 0; shape < 2; ++shape) {
                const int64_t K = shape ? 2048 : 6144, N = shape ? 6144 : 2048;
                const int64_t row_bytes = 4 + (K/256)*68;
                std::vector<uint8_t> hw(size_t(row_bytes)*size_t(N));
                for (auto& b : hw) b = uint8_t(rng());
                { const float sc = 0.01f;
                  for (int64_t r = 0; r < N; ++r) std::memcpy(hw.data()+size_t(r)*row_bytes, &sc, 4); }
                uint8_t* dw = sycl::malloc_device<uint8_t>(hw.size(), dev, ctx);
                q.memcpy(dw, hw.data(), hw.size()).wait();
                for (int T : {16, 32, 64, 128}) {
                    std::vector<float> hx(size_t(T)*size_t(K));
                    for (auto& v : hx) v = float(int(rng()%2001)-1000)/1000.f;
                    std::vector<int32_t> ht(static_cast<size_t>(T), 0);
                    for (int i = 0; i < T; ++i) ht[size_t(i)] = i;
                    float* dx = sycl::malloc_device<float>(hx.size(), dev, ctx);
                    int32_t* dt = sycl::malloc_device<int32_t>(size_t(T), dev, ctx);
                    float* y1 = sycl::malloc_device<float>(size_t(T)*size_t(N), dev, ctx);
                    float* y3 = sycl::malloc_device<float>(size_t(T)*size_t(N), dev, ctx);
                    q.memcpy(dx, hx.data(), hx.size()*4).wait();
                    q.memcpy(dt, ht.data(), size_t(T)*4).wait();
                    const double gf = 2.0*double(K)*double(N)*double(T)/1e9;
                    auto run = [&](const char* nm, auto fn, float* dst) {
                        fn(); fn(); q.wait();
                        auto t0 = Clock::now();
                        for (int r = 0; r < 5; ++r) fn();
                        q.wait();
                        const double ms = dsec(t0)*1e3/5.0;
                        std::printf("    K=%-5lld N=%-5lld T=%-4d %-18s %7.3f ms %7.1f GF/s\n",
                                    (long long)K,(long long)N,T,nm,ms,gf/ms*1e3);
                        (void)dst;
                        return ms;
                    };
                    run("v1", [&]{ k_iq2kt_gemm_v1(q,dw,dx,K,dt,T,y1,K,N,{}); }, y1);
                    run("v3 16x128 4x4 K32", [&]{ k_iq2kt_gemm_v3<16,128,32,4,4>(q,dw,dx,K,dt,T,y3,K,N,{}); }, y3);
                    run("v3 32x128 4x4 K32", [&]{ k_iq2kt_gemm_v3<32,128,32,4,4>(q,dw,dx,K,dt,T,y3,K,N,{}); }, y3);
                    run("v3 16x256 4x4 K32", [&]{ k_iq2kt_gemm_v3<16,256,32,4,4>(q,dw,dx,K,dt,T,y3,K,N,{}); }, y3);
                    run("v3 32x256 4x4 K32", [&]{ k_iq2kt_gemm_v3<32,256,32,4,4>(q,dw,dx,K,dt,T,y3,K,N,{}); }, y3);
                    run("v3 16x128 4x4 K64", [&]{ k_iq2kt_gemm_v3<16,128,64,4,4>(q,dw,dx,K,dt,T,y3,K,N,{}); }, y3);
                    run("v3 32x64  4x4 K64", [&]{ k_iq2kt_gemm_v3<32,64,64,4,4>(q,dw,dx,K,dt,T,y3,K,N,{}); }, y3);
                    {   k_iq2kt_gemm_v1(q,dw,dx,K,dt,T,y1,K,N,{}); q.wait();
                        k_iq2kt_gemm_v3<16,128,32,4,4>(q,dw,dx,K,dt,T,y3,K,N,{}); q.wait();
                        std::vector<float> a(size_t(T)*size_t(N)), b(a.size());
                        q.memcpy(a.data(), y1, a.size()*4).wait();
                        q.memcpy(b.data(), y3, b.size()*4).wait();
                        double mr = 0, rms = 0;
                        for (size_t i=0;i<a.size();++i){ rms += double(a[i])*double(a[i]);
                            const double dd = std::fabs(double(a[i])-double(b[i]));
                            if (dd > mr) mr = dd; }
                        std::printf("      v3 vs v1: max abs diff %.3e (rms %.3e)\n",
                                    mr, std::sqrt(rms/double(a.size())));
                    }
                    sycl::free(dx,ctx); sycl::free(dt,ctx); sycl::free(y1,ctx); sycl::free(y3,ctx);
                }
                sycl::free(dw, ctx);
            }
        }
        {   // Decode-path expert GEMV: one token, TOPK experts. This is 23% of
            // decode time and has no token dimension to amortise the trellis over.
            std::printf("\nDecode expert GEMV (nt=1, top-8 experts)\n");
            std::vector<int8_t> ht2(size_t(65536)*8);
            for (uint32_t q16 = 0; q16 < 65536u; ++q16)
                for (int j = 0; j < 8; ++j)
                    ht2[size_t(q16)*8 + j] = int8_t(trellis_at(q16 + 4096u, j));
            int8_t* dt2 = sycl::malloc_device<int8_t>(ht2.size(), dev, ctx);
            q.memcpy(dt2, ht2.data(), ht2.size()).wait();
            const int64_t K = 6144, N = 2048, rb = 4 + (K/256)*68;
            const int NEE = 8;
            std::vector<uint8_t> hw(size_t(rb)*size_t(N)*size_t(NEE));
            for (auto& b : hw) b = uint8_t(rng());
            for (int e = 0; e < NEE; ++e) { const float sc = 0.006f;
                for (int64_t r = 0; r < N; ++r)
                    std::memcpy(hw.data() + (size_t(e)*size_t(N)+size_t(r))*size_t(rb), &sc, 4); }
            std::vector<float> hx(static_cast<size_t>(K), 0.f);
            for (auto& v : hx) v = float(int(rng()%2001)-1000)/1000.f;
            uint8_t* dw = sycl::malloc_device<uint8_t>(hw.size(), dev, ctx);
            float* dx = sycl::malloc_device<float>(hx.size(), dev, ctx);
            float* y1 = sycl::malloc_device<float>(size_t(NEE)*size_t(N), dev, ctx);
            float* y2 = sycl::malloc_device<float>(size_t(NEE)*size_t(N), dev, ctx);
            std::vector<uint8_t*> hb(static_cast<size_t>(NEE), nullptr);
            for (int e = 0; e < NEE; ++e) hb[size_t(e)] = dw + size_t(e)*size_t(N)*size_t(rb);
            uint8_t** db = sycl::malloc_device<uint8_t*>(size_t(NEE), dev, ctx);
            q.memcpy(dw, hw.data(), hw.size()).wait();
            q.memcpy(dx, hx.data(), hx.size()*4).wait();
            q.memcpy(db, hb.data(), size_t(NEE)*sizeof(uint8_t*)).wait();
            {   // THE architecture test for decode: run the REAL expert GEMV
                // with its weights in HOST memory instead of VRAM. A miss then
                // needs no copy — the kernel streams the weights over PCIe as it
                // computes, which moves identical bytes but removes the copy
                // submission, the dependency, the host wait and the copy/compute
                // stream boundary that run11 identified as the per-block tax.
                // A synthetic uint4 stream hits 26 GB/s; the trellis kernel's
                // access pattern is the open question.
                uint8_t* hw_usm = sycl::malloc_host<uint8_t>(hw.size(), ctx);
                if (hw_usm) {
                    std::memcpy(hw_usm, hw.data(), hw.size());
                    std::vector<uint8_t*> hbh(static_cast<size_t>(NEE), nullptr);
                    for (int e = 0; e < NEE; ++e)
                        hbh[size_t(e)] = hw_usm + size_t(e)*size_t(N)*size_t(rb);
                    uint8_t** dbh = sycl::malloc_device<uint8_t*>(size_t(NEE), dev, ctx);
                    q.memcpy(dbh, hbh.data(), size_t(NEE)*sizeof(uint8_t*)).wait();
                    float* yh = sycl::malloc_device<float>(size_t(NEE)*size_t(N), dev, ctx);
                    const double mb = double(hw.size())/1e6;
                    auto timeit = [&](uint8_t** bases, float* out) {
                        k_iq2kt_gemv_rt<32,8>(q, bases, dx, out, K, N, NEE, {}); q.wait();
                        auto t0 = Clock::now();
                        for (int r = 0; r < 5; ++r) k_iq2kt_gemv_rt<32,8>(q, bases, dx, out, K, N, NEE, {});
                        q.wait();
                        return dsec(t0)/5;
                    };
                    const double tv = timeit(db, y1);
                    const double th = timeit(dbh, yh);
                    // exactness: same weights, same kernel, different residence
                    std::vector<float> a(size_t(NEE)*size_t(N)), b(a.size());
                    q.memcpy(a.data(), y1, a.size()*4).wait();
                    q.memcpy(b.data(), yh, b.size()*4).wait();
                    size_t diff = 0;
                    for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) ++diff;
                    std::printf("  EXPERT GEMV residence: VRAM %6.3f ms (%5.1f GB/s) | "
                                "HOST %6.3f ms (%5.1f GB/s) = %.2fx | %zu/%zu values differ\n",
                                tv*1e3, mb/1e3/tv, th*1e3, mb/1e3/th, th/tv, diff, a.size());
                    std::printf("  -> 8 experts is %.1f MB; a decode block fetches ~2.4 misses "
                                "(%.1f MB) => %.2f ms/block, %.0f ms/token over 78 blocks\n",
                                mb, mb/8*2.4, th*1e3/8*2.4, th*1e3/8*2.4*78);
                    sycl::free(yh, ctx); sycl::free(dbh, ctx); sycl::free(hw_usm, ctx);
                }
            }
            k_iq2kt_gemv<128>(q, db, dx, y1, K, N, NEE, {}, dt2); q.wait();
            std::vector<float> a(size_t(NEE)*size_t(N)), b(a.size());
            q.memcpy(a.data(), y1, a.size()*4).wait();
            auto trial2 = [&](const char* nm, auto fn) {
                fn(); q.wait();
                auto t1 = Clock::now();
                for (int r2 = 0; r2 < 10; ++r2) fn();
                q.wait();
                const double ms = dsec(t1)*1e3/10.0;
                q.memcpy(b.data(), y2, b.size()*4).wait();
                double mx = 0;
                for (size_t i=0;i<a.size();++i){ const double d=std::fabs(double(a[i])-double(b[i]));
                    if (d>mx) mx=d; }
                std::printf("  %-10s %6.3f ms  -> %5.1f ms/token (76 blk x2 mats)  maxabs %.2e\n",
                            nm, ms, ms*152, mx);
            };
            trial2("WG=128", [&]{ k_iq2kt_gemv<128>(q, db, dx, y2, K, N, NEE, {}, dt2); });
            trial2("WG=64",  [&]{ k_iq2kt_gemv<64> (q, db, dx, y2, K, N, NEE, {}, dt2); });
            trial2("WG=32",  [&]{ k_iq2kt_gemv<32> (q, db, dx, y2, K, N, NEE, {}, dt2); });
            trial2("WG=256", [&]{ k_iq2kt_gemv<256>(q, db, dx, y2, K, N, NEE, {}, dt2); });
            trial2("rt 64x2",  [&]{ k_iq2kt_gemv_rt<64,2>(q, db, dx, y2, K, N, NEE, {}); });
            trial2("rt 64x4",  [&]{ k_iq2kt_gemv_rt<64,4>(q, db, dx, y2, K, N, NEE, {}); });
            trial2("rt 64x8",  [&]{ k_iq2kt_gemv_rt<64,8>(q, db, dx, y2, K, N, NEE, {}); });
            trial2("rt 128x4", [&]{ k_iq2kt_gemv_rt<128,4>(q, db, dx, y2, K, N, NEE, {}); });
            trial2("rt 128x8", [&]{ k_iq2kt_gemv_rt<128,8>(q, db, dx, y2, K, N, NEE, {}); });
            trial2("rt 32x4",  [&]{ k_iq2kt_gemv_rt<32,4>(q, db, dx, y2, K, N, NEE, {}); });
            trial2("rt 32x8",  [&]{ k_iq2kt_gemv_rt<32,8>(q, db, dx, y2, K, N, NEE, {}); });
            sycl::free(dw,ctx); sycl::free(dx,ctx); sycl::free(y1,ctx);
            sycl::free(y2,ctx); sycl::free(db,ctx); sycl::free(dt2,ctx);
        }
        {   // Batched MoE GEMM vs the per-expert path it replaces. The previous
            // single-launch attempt was "coherent but wrong", so this compares
            // the packed output element for element before anything runs it.
            // TPE is tokens-per-expert, i.e. chunk/32: what prefill really gives.
            std::printf("\nBatched MoE GEMM (all experts, one launch) vs per-expert\n");
            std::vector<int8_t> htt(size_t(65536)*8);
            for (uint32_t q16 = 0; q16 < 65536u; ++q16)
                for (int j = 0; j < 8; ++j)
                    htt[size_t(q16)*8 + j] = int8_t(trellis_at(q16 + 4096u, j));
            int8_t* dtt = sycl::malloc_device<int8_t>(htt.size(), dev, ctx);
            q.memcpy(dtt, htt.data(), htt.size()).wait();
            const int NEt = 64;
            for (int TPE : {16, 32, 64, 128}) {
            for (int shape = 0; shape < 2; ++shape) {
                const int64_t K = shape ? 2048 : 6144, N = shape ? 6144 : 2048;
                const int64_t rb = 4 + (K/256)*68;
                const int TOK = TPE*NEt/8;
                std::vector<uint8_t> hw(size_t(rb)*size_t(N)*size_t(NEt));
                for (auto& b : hw) b = uint8_t(rng());
                for (int e = 0; e < NEt; ++e) {
                    const float sc = 0.004f + 0.001f*float(e%7);
                    for (int64_t r = 0; r < N; ++r)
                        std::memcpy(hw.data() + (size_t(e)*size_t(N)+size_t(r))*size_t(rb), &sc, 4);
                }
                std::vector<float> hx(size_t(TOK)*size_t(K));
                for (auto& v : hx) v = float(int(rng()%2001)-1000)/1000.f;
                std::vector<std::vector<int32_t>> et{static_cast<size_t>(NEt)};
                for (int t = 0; t < TOK; ++t)
                    for (int k = 0; k < 8; ++k)
                        et[size_t((t*7 + k*11 + (t/3)) % NEt)].push_back(t);
                std::vector<int32_t> halltok, heo, hec, hgo;
                std::vector<uint8_t*> hb;
                uint8_t* dw = sycl::malloc_device<uint8_t>(hw.size(), dev, ctx);
                q.memcpy(dw, hw.data(), hw.size()).wait();
                int rows = 0;
                for (int e = 0; e < NEt; ++e) {
                    const int nt = int(et[size_t(e)].size());
                    if (!nt) continue;
                    hb.push_back(dw + size_t(e)*size_t(N)*size_t(rb));
                    heo.push_back(int32_t(halltok.size()));
                    hec.push_back(nt); hgo.push_back(rows);
                    for (auto v : et[size_t(e)]) halltok.push_back(v);
                    rows += nt;
                }
                const int ne = int(hb.size());
                float* dx = sycl::malloc_device<float>(hx.size(), dev, ctx);
                int32_t* dtok = sycl::malloc_device<int32_t>(halltok.size(), dev, ctx);
                uint8_t** db = sycl::malloc_device<uint8_t*>(size_t(ne), dev, ctx);
                int32_t* deo = sycl::malloc_device<int32_t>(size_t(ne), dev, ctx);
                int32_t* dec = sycl::malloc_device<int32_t>(size_t(ne), dev, ctx);
                int32_t* dgo = sycl::malloc_device<int32_t>(size_t(ne), dev, ctx);
                int32_t* dtl = sycl::malloc_device<int32_t>(size_t(rows)*2 + 2*size_t(ne), dev, ctx);
                float* ya = sycl::malloc_device<float>(size_t(rows)*size_t(N), dev, ctx);
                // +64 rows of pad: the XMX store writes whole 8-row tiles.
                float* yb = sycl::malloc_device<float>((size_t(rows)+64)*size_t(N), dev, ctx);
                q.memcpy(dx, hx.data(), hx.size()*4).wait();
                q.memcpy(dtok, halltok.data(), halltok.size()*4).wait();
                q.memcpy(db, hb.data(), size_t(ne)*sizeof(uint8_t*)).wait();
                q.memcpy(deo, heo.data(), size_t(ne)*4).wait();
                q.memcpy(dec, hec.data(), size_t(ne)*4).wait();
                q.memcpy(dgo, hgo.data(), size_t(ne)*4).wait();
                auto per_expert = [&]{
                    for (int k = 0; k < ne; ++k)
                        k_iq2kt_gemm_v1(q, hb[size_t(k)], dx, K, dtok + heo[size_t(k)],
                                        hec[size_t(k)], ya + int64_t(hgo[size_t(k)])*N, K, N, {});
                    q.wait();
                };
                per_expert();
                auto t0 = Clock::now(); for (int r=0;r<3;++r) per_expert();
                const double m1 = dsec(t0)*1e3/3.0;
                const double gf = 2.0*double(K)*double(N)*double(rows)/1e9;
                std::vector<float> a(size_t(rows)*size_t(N)), b(a.size());
                q.memcpy(a.data(), ya, a.size()*4).wait();
                double rms = 0;
                for (size_t z = 0; z < a.size(); ++z) rms += double(a[z])*double(a[z]);
                rms = std::sqrt(rms/double(a.size()));
                std::printf("  TPE=%-4d K=%-5lld N=%-5lld %d rows | per-expert %7.2f ms (%7.1f GF/s)\n",
                            TPE, (long long)K, (long long)N, rows, m1, gf/m1*1e3);
                auto trial = [&](const char* nm, int BMv, auto fn) {
                    std::vector<int32_t> htl;
                    for (int k = 0; k < ne; ++k)
                        for (int m = 0; m < hec[size_t(k)]; m += BMv) { htl.push_back(k); htl.push_back(m); }
                    const int ntiles = int(htl.size()/2);
                    q.memcpy(dtl, htl.data(), htl.size()*4).wait();
                    fn(ntiles);
                    auto t1 = Clock::now(); for (int r=0;r<3;++r) fn(ntiles);
                    const double m2 = dsec(t1)*1e3/3.0;
                    q.memcpy(b.data(), yb, b.size()*4).wait();
                    double mx = 0;
                    for (size_t z = 0; z < a.size(); ++z) {
                        const double d = std::fabs(double(a[z]) - double(b[z]));
                        if (d > mx) mx = d;
                    }
                    std::printf("      %-14s %5d tiles %7.2f ms (%7.1f GF/s) %5.2fx  maxabs %.2e / rms %.2e\n",
                                nm, ntiles, m2, gf/m2*1e3, m1/m2, mx, rms);
                };
                trial("16x128 K32", 16, [&](int nt2){ k_iq2kt_moe_gemm<16,128,32,4,4>(q,dtt,db,deo,dec,dgo,dtl,nt2,0,dtok,dx,K,yb,K,N,{}); q.wait(); });
                trial("32x128 K32", 32, [&](int nt2){ k_iq2kt_moe_gemm<32,128,32,4,4>(q,dtt,db,deo,dec,dgo,dtl,nt2,0,dtok,dx,K,yb,K,N,{}); q.wait(); });
                trial("64x128 K32", 64, [&](int nt2){ k_iq2kt_moe_gemm<64,128,32,4,4>(q,dtt,db,deo,dec,dgo,dtl,nt2,0,dtok,dx,K,yb,K,N,{}); q.wait(); });
                trial("64x256 K32", 64, [&](int nt2){ k_iq2kt_moe_gemm<64,256,32,4,4>(q,dtt,db,deo,dec,dgo,dtl,nt2,0,dtok,dx,K,yb,K,N,{}); q.wait(); });
                trial("32x256 K32", 32, [&](int nt2){ k_iq2kt_moe_gemm<32,256,32,4,4>(q,dtt,db,deo,dec,dgo,dtl,nt2,0,dtok,dx,K,yb,K,N,{}); q.wait(); });
                trial("64x64  K32", 64, [&](int nt2){ k_iq2kt_moe_gemm<64,64,32,4,4>(q,dtt,db,deo,dec,dgo,dtl,nt2,0,dtok,dx,K,yb,K,N,{}); q.wait(); });
                trial("64x128 K64", 64, [&](int nt2){ k_iq2kt_moe_gemm<64,128,64,4,4>(q,dtt,db,deo,dec,dgo,dtl,nt2,0,dtok,dx,K,yb,K,N,{}); q.wait(); });
                trial("32x128 K64", 32, [&](int nt2){ k_iq2kt_moe_gemm<32,128,64,4,4>(q,dtt,db,deo,dec,dgo,dtl,nt2,0,dtok,dx,K,yb,K,N,{}); q.wait(); });
                trial("grf 64x256 8x8", 64, [&](int nt2){ k_iq2kt_moe_gemm<64,256,32,8,8,true>(q,dtt,db,deo,dec,dgo,dtl,nt2,0,dtok,dx,K,yb,K,N,{}); q.wait(); });
                trial("grf 32x256 8x8", 32, [&](int nt2){ k_iq2kt_moe_gemm<32,256,32,8,8,true>(q,dtt,db,deo,dec,dgo,dtl,nt2,0,dtok,dx,K,yb,K,N,{}); q.wait(); });
                trial("grf 64x128 8x8", 64, [&](int nt2){ k_iq2kt_moe_gemm<64,128,32,8,8,true>(q,dtt,db,deo,dec,dgo,dtl,nt2,0,dtok,dx,K,yb,K,N,{}); q.wait(); });
                trial("grf 64x256 4x4", 64, [&](int nt2){ k_iq2kt_moe_gemm<64,256,32,4,4,true>(q,dtt,db,deo,dec,dgo,dtl,nt2,0,dtok,dx,K,yb,K,N,{}); q.wait(); });
                trial("grf 32x256 4x8", 32, [&](int nt2){ k_iq2kt_moe_gemm<32,256,32,4,8,true>(q,dtt,db,deo,dec,dgo,dtl,nt2,0,dtok,dx,K,yb,K,N,{}); q.wait(); });
                trial("grf 64x512 8x8", 64, [&](int nt2){ k_iq2kt_moe_gemm<64,512,32,8,8,true>(q,dtt,db,deo,dec,dgo,dtl,nt2,0,dtok,dx,K,yb,K,N,{}); q.wait(); });
                trial("128x256 8x8",   128, [&](int nt2){ k_iq2kt_moe_gemm<128,256,32,8,8,true>(q,dtt,db,deo,dec,dgo,dtl,nt2,0,dtok,dx,K,yb,K,N,{}); q.wait(); });
                trial("grf 64x256 8x4", 64, [&](int nt2){ k_iq2kt_moe_gemm<64,256,32,8,4,true>(q,dtt,db,deo,dec,dgo,dtl,nt2,0,dtok,dx,K,yb,K,N,{}); q.wait(); });
                std::printf("      -- fused gate+up (counts BOTH matrices' flops) --\n");
                trial("fuse 64x128 4x4", 64, [&](int nt2){ k_iq2kt_moe_gemm2<64,128,32,4,4,true>(q,dtt,db,deo,dec,dgo,dtl,nt2,0,0,dtok,dx,K,yb,ya,K,N,{}); q.wait(); });
                trial("fuse 64x256 4x4", 64, [&](int nt2){ k_iq2kt_moe_gemm2<64,256,32,4,4,true>(q,dtt,db,deo,dec,dgo,dtl,nt2,0,0,dtok,dx,K,yb,ya,K,N,{}); q.wait(); });
                trial("fuse 32x256 4x4", 32, [&](int nt2){ k_iq2kt_moe_gemm2<32,256,32,4,4,true>(q,dtt,db,deo,dec,dgo,dtl,nt2,0,0,dtok,dx,K,yb,ya,K,N,{}); q.wait(); });
                trial("fuse 64x128 8x4", 64, [&](int nt2){ k_iq2kt_moe_gemm2<64,128,32,8,4,true>(q,dtt,db,deo,dec,dgo,dtl,nt2,0,0,dtok,dx,K,yb,ya,K,N,{}); q.wait(); });
                std::printf("      -- XMX/DPAS (fp16 operands, fp32 accumulate) --\n");
                trial("xmx 64x128 K128", 64, [&](int nt2){ k_iq2kt_moe_xmx<64,128,128>(q,db,deo,dec,dgo,dtl,nt2,0,dtok,dx,K,yb,K,N,{}); q.wait(); });
                trial("xmx 32x128 K128", 32, [&](int nt2){ k_iq2kt_moe_xmx<32,128,128>(q,db,deo,dec,dgo,dtl,nt2,0,dtok,dx,K,yb,K,N,{}); q.wait(); });
                trial("xmx 64x128 K64",  64, [&](int nt2){ k_iq2kt_moe_xmx<64,128,64>(q,db,deo,dec,dgo,dtl,nt2,0,dtok,dx,K,yb,K,N,{}); q.wait(); });
                trial("xmx 64x256 K64",  64, [&](int nt2){ k_iq2kt_moe_xmx<64,256,64>(q,db,deo,dec,dgo,dtl,nt2,0,dtok,dx,K,yb,K,N,{}); q.wait(); });
                trial("xmx 32x256 K64",  32, [&](int nt2){ k_iq2kt_moe_xmx<32,256,64>(q,db,deo,dec,dgo,dtl,nt2,0,dtok,dx,K,yb,K,N,{}); q.wait(); });
                trial("xmx 64x256 K128", 64, [&](int nt2){ k_iq2kt_moe_xmx<64,256,128>(q,db,deo,dec,dgo,dtl,nt2,0,dtok,dx,K,yb,K,N,{}); q.wait(); });
                sycl::free(dw,ctx); sycl::free(dx,ctx); sycl::free(dtok,ctx);
                sycl::free(db,ctx); sycl::free(deo,ctx); sycl::free(dec,ctx);
                sycl::free(dgo,ctx); sycl::free(dtl,ctx);
                sycl::free(ya,ctx); sycl::free(yb,ctx);
            }
            }
        }
        std::printf("\n");
        return 0;
    }

    // ---- trellis lookup table ----------------------------------------------
    // At DECODE a routed expert's weight is decoded once and used for exactly one
    // token, so the trellis is ~90% of the expert GEMV's work: 8 multiplies, 8
    // masks, 16 shift/adds and 8 subtracts per group of 8 weights. The value
    // depends only on the 16-bit ql16 and the lane, so all 65536x8 of them fit
    // in a 512 KiB table -- one 8-byte L1/L2 load per group instead of ~64 ops.
    std::vector<int8_t> h_ttab(size_t(65536)*8);
    for (uint32_t q16 = 0; q16 < 65536u; ++q16)
        for (int j = 0; j < 8; ++j)
            h_ttab[size_t(q16)*8 + j] = int8_t(trellis_at(q16 + 4096u, j));
    std::vector<int8_t*> TT(size_t(NG), nullptr);
    for (int d = 0; d < NG; ++d) {
        TT[size_t(d)] = sycl::malloc_device<int8_t>(h_ttab.size(), devs[d], ctxs[d]);
        sycl::queue(ctxs[d], devs[d]).memcpy(TT[size_t(d)], h_ttab.data(), h_ttab.size()).wait();
    }
    std::printf("trellis table: %.2f MiB per card\n", h_ttab.size()/1048576.0);

    // ---- upload helper: Q6_0 gets SoA-repacked, others go verbatim ----------
    uint64_t vram_used = 0;
    std::vector<uint8_t> stage;
    auto upload = [&](const char* name, bool required) -> QW {
        QW w;
        const auto* ti = g.find_tensor(name);
        if (!ti) { if (required) { std::fprintf(stderr,"missing tensor %s\n", name); std::exit(1);} return w; }
        w.dt = ti->dtype;
        w.K = int64_t(ti->shape[0]);
        w.N = 1; for (uint32_t d=1; d<ti->n_dims; ++d) w.N *= int64_t(ti->shape[d]);
        if (ti->dtype == DType::kQ6_0) {
            w.row_bytes = q6_soa_row_bytes(w.K);
            const size_t total = size_t(w.row_bytes)*size_t(w.N);
            stage.resize(total);
            const int64_t src_row = (w.K/32)*26;
            for (int64_t r = 0; r < w.N; ++r)
                q6_repack_row(ti->data + r*src_row, stage.data() + r*w.row_bytes, w.K);
            w.dev = sycl::malloc_device<uint8_t>(total, dev, ctx);
            q.memcpy(w.dev, stage.data(), total).wait();
            vram_used += total;
        } else {
            w.row_bytes = int64_t(ti->nbytes) / w.N;
            w.dev = sycl::malloc_device<uint8_t>(ti->nbytes, dev, ctx);
            q.memcpy(w.dev, ti->data, ti->nbytes).wait();
            vram_used += ti->nbytes;
        }
        return w;
    };

    // Q6_0 row range [r0, r1) onto device d1. Used for the per-head splits.
    uint64_t vram_used1 = 0;
    sycl::queue q1 = (NG > 1) ? sycl::queue(ctxs[1], devs[1], qprops())
                              : sycl::queue(ctxs[0], devs[0], qprops());
    // Q6_0 with every row narrowed to input range [k0, k1), onto device dd.
    // wo splits along K, not N: card d owns heads [32d, 32d+32), which are
    // exactly wo's input columns [8192d, 8192d+8192). Each SoA section is a
    // contiguous slice because k0/k1 are multiples of the 32-weight block.
    auto upload_kslice = [&](const char* name, int64_t k0, int64_t k1, int dd) -> QW {
        QW w;
        const auto* ti = g.find_tensor(name);
        if (!ti) { std::fprintf(stderr,"missing tensor %s\n", name); std::exit(1); }
        const int64_t Kfull = int64_t(ti->shape[0]);
        w.dt = ti->dtype;
        w.K = k1 - k0;
        w.N = 1; for (uint32_t d2=1; d2<ti->n_dims; ++d2) w.N *= int64_t(ti->shape[d2]);
        w.row_bytes = q6_soa_row_bytes(w.K);
        const size_t total = size_t(w.row_bytes)*size_t(w.N);
        stage.resize(total);
        std::vector<uint8_t> full(size_t(q6_soa_row_bytes(Kfull)));
        const int64_t src_row = (Kfull/32)*26;
        const int64_t nbF = Kfull/32, b0 = k0/32, nbS = w.K/32;
        for (int64_t r = 0; r < w.N; ++r) {
            q6_repack_row(ti->data + r*src_row, full.data(), Kfull);
            uint8_t* dst = stage.data() + r*w.row_bytes;
            std::memcpy(dst,                 full.data() + b0*2,                   size_t(nbS)*2);
            std::memcpy(dst + nbS*2,         full.data() + nbF*2 + b0*16,          size_t(nbS)*16);
            std::memcpy(dst + nbS*2 + w.K/2, full.data() + nbF*2 + Kfull/2 + b0*8, size_t(nbS)*8);
        }
        w.dev = sycl::malloc_device<uint8_t>(total, devs[dd], ctxs[dd]);
        (dd == 0 ? q : q1).memcpy(w.dev, stage.data(), total).wait();
        if (dd == 0) vram_used += total; else vram_used1 += total;
        return w;
    };
    auto upload_rows = [&](const char* name, int64_t r0, int64_t r1) -> QW {
        QW w;
        const auto* ti = g.find_tensor(name);
        if (!ti) { std::fprintf(stderr,"missing tensor %s\n", name); std::exit(1); }
        w.dt = ti->dtype;
        w.K = int64_t(ti->shape[0]);
        w.N = r1 - r0;
        w.row_bytes = q6_soa_row_bytes(w.K);
        const size_t total = size_t(w.row_bytes)*size_t(w.N);
        stage.resize(total);
        const int64_t src_row = (w.K/32)*26;
        for (int64_t r = 0; r < w.N; ++r)
            q6_repack_row(ti->data + (r0 + r)*src_row, stage.data() + r*w.row_bytes, w.K);
        w.dev = sycl::malloc_device<uint8_t>(total, devs[1], ctxs[1]);
        q1.memcpy(w.dev, stage.data(), total).wait();
        vram_used1 += total;
        return w;
    };
    auto upload_f32 = [&](const char* name) -> float* {
        const auto* ti = g.find_tensor(name);
        if (!ti) { std::fprintf(stderr,"missing %s\n", name); std::exit(1); }
        float* d = sycl::malloc_device<float>(ti->nbytes/4, dev, ctx);
        q.memcpy(d, ti->data, ti->nbytes).wait();
        vram_used += ti->nbytes;
        return d;
    };
    auto host_ptr = [&](const char* name) -> const uint8_t* {
        const auto* ti = g.find_tensor(name);
        if (!ti) { std::fprintf(stderr,"missing %s\n", name); std::exit(1); }
        return ti->data;
    };

    auto t_load = Clock::now();
    char nm[128];
    QW tok_embd = upload("token_embd.weight", true);
    QW lm_head  = upload("output.weight", true);
    float* out_norm = upload_f32("output_norm.weight");

    // NL transformer blocks plus, at index NL, the NextN/MTP block. The MTP
    // block is a COMPLETE MLA + 256-expert MoE layer with its own expert bank —
    // running it costs about 1/78th of a full forward, which is what makes
    // speculative drafting nearly free.
    const bool want_mtp = getenv("IE_GLM52_MTP") != nullptr;
    // IE_GLM52_TP=1: split attention across both cards by head.
    const bool want_tp  = (NG > 1) && getenv("IE_GLM52_TP") != nullptr;
    // Splitting the shared expert as well produced degenerate output alongside
    // the group ramp; off by default until it is isolated. It was worth ~0.55 s.
    if (getenv("IE_GLM52_SHTP")) g_shtp = atoi(getenv("IE_GLM52_SHTP"));
    // weights for BOTH layouts are uploaded when TP is on, so this can be swept
    // per config from one load.
    const bool have_shtp = want_tp;
    #define want_shtp (have_shtp && g_shtp)
    std::vector<Layer> L(NL + (want_mtp ? 1 : 0));
    for (int il = 0; il < NL + (want_mtp ? 1 : 0); ++il) {
        Layer& l = L[il];
        auto nn = [&](const char* s){ std::snprintf(nm,sizeof(nm),"blk.%d.%s",il,s); return nm; };
        l.attn_norm = upload_f32(nn("attn_norm.weight"));
        l.q_a_norm  = upload_f32(nn("attn_q_a_norm.weight"));
        l.kv_a_norm = upload_f32(nn("attn_kv_a_norm.weight"));
        l.ffn_norm  = upload_f32(nn("ffn_norm.weight"));
        l.wq_a  = upload(nn("attn_q_a.weight"), true);
        l.wq_b  = upload(nn("attn_q_b.weight"), true);
        l.wkv_a = upload(nn("attn_kv_a_mqa.weight"), true);
        l.wk_b  = upload(nn("attn_k_b.weight"), true);
        l.wv_b  = upload(nn("attn_v_b.weight"), true);
        if (!want_tp) l.wo = upload(nn("attn_output.weight"), true);
        if (want_tp) {                       // GPU1's half: heads [NH/2, NH)
            const int64_t hh = NH/2;
            l.wq_b1 = upload_rows(nn("attn_q_b.weight"), hh*C.key_len_mla, int64_t(NH)*C.key_len_mla);
            l.wk_b1 = upload_rows(nn("attn_k_b.weight"), hh*KVL, int64_t(NH)*KVL);
            l.wv_b1 = upload_rows(nn("attn_v_b.weight"), hh*VHD, int64_t(NH)*VHD);
            // wo is stored on GPU0 as its TWO K-slices rather than whole: prefill
            // uses the first, decode sums both, and the total is the same 6.4 GB
            // the full tensor cost -- so the split takes no VRAM from the expert
            // cache (holding both cost gpu0 ~330 slots and 1.6 s of prefill).
            l.wo0h  = upload_kslice(nn("attn_output.weight"), 0,      hh*VHD, 0);
            l.wo0b  = upload_kslice(nn("attn_output.weight"), hh*VHD, int64_t(NH)*VHD, 0);
            l.wo1h  = upload_kslice(nn("attn_output.weight"), hh*VHD, int64_t(NH)*VHD, 1);
        }
        if (C.is_dense_layer(uint32_t(il))) {
            l.ffn_gate = upload(nn("ffn_gate.weight"), true);
            l.ffn_up   = upload(nn("ffn_up.weight"), true);
            l.ffn_down = upload(nn("ffn_down.weight"), true);
        } else {
            l.router      = upload(nn("ffn_gate_inp.weight"), true);
            l.exp_probs_b = upload_f32(nn("exp_probs_b.bias"));
            {   const auto* bt = g.find_tensor(nn("exp_probs_b.bias"));
                l.h_bias.assign(reinterpret_cast<const float*>(bt->data),
                                reinterpret_cast<const float*>(bt->data) + NE); }
            l.sh_gate = upload(nn("ffn_gate_shexp.weight"), true);
            l.sh_up   = upload(nn("ffn_up_shexp.weight"), true);
            l.sh_down = upload(nn("ffn_down_shexp.weight"), true);
            // Upload the split layout whenever TP is on, not only when the split
            // is enabled: the flag is swept per config from one load, and a
            // config that turns it on without the weights present runs on
            // unloaded pointers — fast, and producing garbage (measured: 107.52
            // "tok/s" emitting " Dodd gods gods...").
            // The split layout is a SECOND copy of the shared expert (~2 GiB
            // across both cards = ~208 expert slots). It is measured worth
            // nothing (§25g) and off by default, so upload it only when asked —
            // and when sweeping the flag, force it on at load time.
            // A second, unused copy of the shared expert costs ~2 GiB = ~208
            // expert slots. The split is measured worth nothing (§25g), so this
            // is uploaded only on request; sweeps must force it with
            // IE_GLM52_SHTP_LOAD so the flag never runs on unloaded pointers.
            if (have_shtp && (g_shtp || getenv("IE_GLM52_SHTP_LOAD"))) {
                // sh_gate/sh_up split by output row, so GPU0 just uses a prefix of
                // the full tensors and needs no extra upload. sh_down splits along
                // K and is stored as its two slices, same total bytes as whole.
                const int64_t eh = EF/2;
                l.shg1  = upload_rows(nn("ffn_gate_shexp.weight"), eh, EF);
                l.shu1  = upload_rows(nn("ffn_up_shexp.weight"),   eh, EF);
                l.shd0  = upload_kslice(nn("ffn_down_shexp.weight"), 0,  eh, 0);
                l.shd0b = upload_kslice(nn("ffn_down_shexp.weight"), eh, EF, 0);
                l.shd1  = upload_kslice(nn("ffn_down_shexp.weight"), eh, EF, 1);
            }
            l.host_gate = host_ptr(nn("ffn_gate_exps.weight"));
            l.host_up   = host_ptr(nn("ffn_up_exps.weight"));
            l.host_down = host_ptr(nn("ffn_down_exps.weight"));
            {   const uint64_t base = g.tensor_data_offset();
                l.off_gate = base + g.find_tensor(nn("ffn_gate_exps.weight"))->offset_in_data;
                l.off_up   = base + g.find_tensor(nn("ffn_up_exps.weight"))->offset_in_data;
                l.off_down = base + g.find_tensor(nn("ffn_down_exps.weight"))->offset_in_data; }
        }
        if ((il % 10) == 0)
            std::printf("  loaded block %2d/%d   VRAM %.2f GiB   %.0fs\n",
                        il, NL, vram_used/1073741824.0, dsec(t_load));
    }
    float *mtp_enorm=nullptr, *mtp_hnorm=nullptr, *mtp_shnorm=nullptr;
    QW mtp_eh;
    if (want_mtp) {
        std::snprintf(nm,sizeof(nm),"blk.%d.nextn.enorm.weight",NL);            mtp_enorm  = upload_f32(nm);
        std::snprintf(nm,sizeof(nm),"blk.%d.nextn.hnorm.weight",NL);            mtp_hnorm  = upload_f32(nm);
        std::snprintf(nm,sizeof(nm),"blk.%d.nextn.shared_head_norm.weight",NL); mtp_shnorm = upload_f32(nm);
        std::snprintf(nm,sizeof(nm),"blk.%d.nextn.eh_proj.weight",NL);          mtp_eh     = upload(nm, true);
        std::printf("MTP block %d loaded (eh_proj %lldx%lld)\n",
                    NL, (long long)mtp_eh.K, (long long)mtp_eh.N);
    }
    stage.clear(); stage.shrink_to_fit();
    std::printf("weights in VRAM: %.2f GiB in %.0f s\n\n",
                vram_used/1073741824.0, dsec(t_load));

    // ---- pin the routed-expert bank ---------------------------------------
    // Fetching straight from the mmap'd GGUF measured 10.1 GB/s against a 26.6
    // GB/s link: the driver has to pin and release every non-registered source
    // region on each transfer. Copying the bank into host memory that is pinned
    // ONCE removes that per-transfer cost. 208 GiB of pinned host memory is
    // allocatable here (max single allocation 30.3 GiB), and the bank is 182.4
    // GiB, so all of it fits; anything that does not is left on mmap.
    const uint64_t pin_budget = uint64_t(getenv("IE_GLM52_PIN_GIB")
                                         ? atoll(getenv("IE_GLM52_PIN_GIB")) : 170) << 30;
    uint64_t pinned_bytes = 0;
    int pinned_layers = 0;
    {
        auto t_pin = Clock::now();
        // Copy in FILE ORDER so the reads stay sequential — the source is a
        // 196 GiB file on a 120 MB/s spinning disk, and any page the cache has
        // dropped is re-read from it.
        struct Job { const uint8_t* src; uint64_t bytes; const uint8_t** dst_slot; };
        std::vector<std::pair<uint64_t, Job>> jobs;
        for (int il = 0; il < NL; ++il) {
            if (C.is_dense_layer(uint32_t(il))) continue;
            auto nn2 = [&](const char* s){ std::snprintf(nm,sizeof(nm),"blk.%d.%s",il,s); return nm; };
            for (auto [suf, slotp] : { std::pair<const char*, const uint8_t**>{"ffn_gate_exps.weight", &L[il].host_gate},
                                       {"ffn_up_exps.weight",   &L[il].host_up},
                                       {"ffn_down_exps.weight", &L[il].host_down} }) {
                const auto* ti = g.find_tensor(nn2(suf));
                jobs.push_back({ ti->offset_in_data, Job{ ti->data, ti->nbytes, slotp } });
            }
        }
        std::sort(jobs.begin(), jobs.end(),
                  [](auto& a, auto& b){ return a.first < b.first; });
        for (auto& [off, j] : jobs) {
            if (pinned_bytes + j.bytes > pin_budget) continue;   // leave on mmap
            void* p = sycl::malloc_host(j.bytes, ctx);
            if (!p) continue;
            std::memcpy(p, j.src, j.bytes);
            *j.dst_slot = static_cast<const uint8_t*>(p);
            pinned_bytes += j.bytes;
            if ((++pinned_layers % 30) == 0)
                std::printf("  pinned %.1f GiB (%.0f s, %.0f MB/s)\n",
                            pinned_bytes/1073741824.0, dsec(t_pin),
                            pinned_bytes/dsec(t_pin)/1e6);
        }
        std::printf("expert bank pinned: %.2f GiB of %.2f GiB in %.0f s\n\n",
                    pinned_bytes/1073741824.0,
                    182.43, dsec(t_pin));
    }

    // ---- expert slot cache -------------------------------------------------
    const int64_t g_row = 4 + (int64_t(H)/256)*68,  g_rows = EF;      // gate/up
    const int64_t d_row = 4 + (int64_t(EF)/256)*68, d_rows = H;       // down
    const size_t  gu_bytes = size_t(g_row)*g_rows;
    const size_t  dn_bytes = size_t(d_row)*d_rows;
    const size_t  slot_bytes = gu_bytes*2 + dn_bytes;
    // One independent LRU cache per GPU. GPU1 stores no resident weights, so it
    // gets far more slots — sized from what is actually free on each card.
    struct Cache {
        sycl::queue qc, qx;                 // compute + transfer, on its device
        // Decode issues ~1.4 expert transfers per block and waits on them, so a
        // single in-order transfer queue serialises their LATENCY as well as
        // their bytes: the same host tier sustains 22.8 GB/s per card under
        // prefill's deep queue and 13.8 under decode's. Several queues let those
        // latencies overlap.
        std::vector<sycl::queue> qxs;
        size_t qxi = 0;
        std::vector<uint8_t*> slot;
        std::vector<int64_t>  key;          // slot -> layer*NE+expert
        std::vector<uint64_t> age;
        std::vector<int>      lookup;       // layer*NE+expert -> slot
        uint64_t hits=0, misses=0, h2d=0;
        // Pinned staging ring: pread lands here, then one DMA to the slot.
        std::vector<uint8_t*>    stage;
        std::vector<sycl::event> stage_ev;
        size_t                   ring = 0;
        // The DMA that last filled each slot. A PREFETCHED slot looks like a hit
        // to the next block, so its fill event must still gate the kernel that
        // reads it; waiting on an already-complete event is free.
        std::vector<sycl::event> fill;
        uint64_t pin_hits = 0;              // misses served from the shared hot tier
        uint64_t disk_reads = 0;            // misses that fell through to pread()
        double   disk_secs = 0;             // ...and what they cost the host
        uint64_t pf_issued = 0, pf_used = 0;    // predictive prefetch
        uint64_t nk_total = 0, nk_blocks = 0;   // expert-count balance across cards
        size_t   static_cap = 0;                // slots reserved for static placement
        size_t hand = 0;                    // CLOCK eviction hand
        // Last slot-reading kernel of the previous block on this card. With the
        // top-of-block drain gone (g_rp), the first eviction copy of a block
        // must be ordered after it — a qx barrier, not a host wait.
        sycl::event last_moe;
    };
    // Per-device pinned hot-tier budget. Two cards x 24 GiB leaves ~200 GiB of
    // RAM free — the 175 GiB single pin that preceded this left ~8 GiB and took
    // the desktop down with it.
    const uint64_t pin_tier_bytes = uint64_t(getenv("IE_GLM52_HOT_GIB")
                                             ? atoll(getenv("IE_GLM52_HOT_GIB")) : 24) << 30;
    // ---- shared hot tier --------------------------------------------------
    // Measured DMA source rates on this box: mmap file-backed 11.64 GB/s,
    // anonymous malloc 11.92, malloc_host 26.10, and anonymous memory passed
    // through prepare_for_device_copy 25.95. Registration is a NO-OP for
    // file-backed pages but doubles anonymous memory — so the tier is one plain
    // anonymous buffer registered with BOTH contexts. That means a single copy
    // serves both cards (per-device pinned tiers duplicated every shared expert)
    // and, because the file pages it replaces are dropped with fadvise, it does
    // not compete with the page cache the way pinned USM did.
    const uint64_t hot_cap = uint64_t(getenv("IE_GLM52_HOT_GIB")
                                      ? atoll(getenv("IE_GLM52_HOT_GIB")) : 56) << 30;
    uint8_t* hot_base = nullptr;
    uint64_t hot_used = 0;
    // Under IE_GLM52_STREAM the tier must be USM the GPU can dereference, not
    // plain anonymous memory: a kernel reading registered anonymous memory
    // measured 502 GB/s, i.e. the pages had migrated into VRAM, which is not a
    // behaviour to rely on across a 147 GiB tier. malloc_host stays put.
    std::vector<uint8_t*> hot_chunks;
    uint64_t hot_chunk_cap = 0, hot_chunk_used = 0;
    auto hot_alloc = [&](uint64_t need) -> uint8_t* {
        if (!g_stream) {                       // one anonymous, huge-paged arena
            if (hot_used + need > hot_cap) return nullptr;
            uint8_t* p2 = hot_base + hot_used;
            hot_used += need;
            return p2;
        }
        if (hot_chunks.empty() || hot_chunk_used + need > hot_chunk_cap) {
            if (hot_used + need > hot_cap) return nullptr;
            const uint64_t want = std::min<uint64_t>(4ull<<30, hot_cap - hot_used);
            if (want < need) return nullptr;
            uint8_t* ch = sycl::malloc_host<uint8_t>(size_t(want), ctxs[0]);
            if (!ch) return nullptr;
            // give the second card's COPY path the same fast registered access
            if (NG > 1)
                sycl::ext::oneapi::experimental::prepare_for_device_copy(ch, size_t(want), ctxs[1]);
            hot_chunks.push_back(ch);
            hot_chunk_cap = want; hot_chunk_used = 0;
        }
        uint8_t* p2 = hot_chunks.back() + hot_chunk_used;
        hot_chunk_used += need; hot_used += need;
        return p2;
    };
    std::vector<uint8_t*> hot_ptr;
    // ---- profile-guided static expert placement ---------------------------
    // The LRU caches measured 49.1%/42.8% hit at 4,459 slots. The router trace
    // says the 4,459 HOTTEST (block,expert) pairs cover 85.4% of all requests.
    // LRU is losing most of that because reuse distance (p50 ~2,396 requests) is
    // far longer than a cache this size can bridge, so it thrashes.
    //
    // So: rank experts by observed frequency, split ownership between the cards
    // alternately down that ranking (equal count AND equal hotness, no expert
    // duplicated), and make each card's share permanently resident. Everything
    // else stays in the host tier and is demand-fetched.
    // Percentage of experts owned by GPU0. Default 50; IE_GLM52_SPLIT0 overrides.
    const int SPLIT0 = getenv("IE_GLM52_SPLIT0") ? atoi(getenv("IE_GLM52_SPLIT0")) : 50;
    const int NLA = NL + (want_mtp ? 1 : 0);      // blocks incl. MTP
    std::vector<int8_t> owner(size_t(NLA)*NE, -1);
    std::vector<int32_t> rank(size_t(NLA)*NE, INT32_MAX);
    {
        const char* pf = getenv("IE_GLM52_PLACEMENT");
        int nread = 0;
        if (pf) {
            FILE* f = fopen(pf, "r");
            if (f) {
                int b, e; long long cnt;
                while (fscanf(f, "%d %d %lld", &b, &e, &cnt) == 3) {
                    if (b < 0 || b >= NLA || e < 0 || e >= NE) continue;
                    const size_t key = size_t(b)*NE + e;
                    if (owner[key] >= 0) continue;
                    // Ownership is split by CACHE CAPACITY, not evenly. GPU0
                    // carries 13.96 GiB of resident weights, so it has room for
                    // ~1,485 expert slots against GPU1's 2,974; splitting the
                    // experts 50/50 leaves GPU0 with half the requests and a
                    // quarter of the demand slots, and its hit rate collapses.
                    owner[key] = int8_t((nread % 100) < SPLIT0 ? 0 : (NG > 1 ? 1 : 0));
                    rank[key]  = nread++;
                }
                fclose(f);
            }
        }
        // Anything the trace never saw: spread by expert id so the split stays
        // balanced for prompts that stray outside the calibration set.
        for (int il = 0; il < NLA; ++il)
            for (int e = 0; e < NE; ++e) {
                const size_t key = size_t(il)*NE + e;
                if (owner[key] < 0)
                    owner[key] = int8_t(((il*NE + e) % 100) < SPLIT0 ? 0 : (NG > 1 ? 1 : 0));
            }
        std::printf("placement: %d ranked (block,expert) pairs from %s\n",
                    nread, pf ? pf : "(none — falling back to expert-id split)");
    }

    const int STAGE_RING = 8;
    const int PREAD_THREADS = 8;
    PreadPool pool(PREAD_THREADS);
    const int gfd = open(path, O_RDONLY);
    if (gfd < 0) { std::fprintf(stderr, "open(gguf) for pread failed\n"); return 1; }

    if (hot_cap) {
        // 2 MiB-aligned + MADV_HUGEPAGE. At 4 KiB pages a 150 GiB tier is ~39M
        // page-table entries, and scattered DMA across it collapsed effective
        // bandwidth from 31 GB/s (110 GiB tier) to 9.6 GB/s even though it was
        // moving FEWER bytes. Huge pages cut the entry count 512x.
        // Two tier kinds, and the choice is NOT free. malloc_host USM is
        // kernel-dereferenceable (required for IE_GLM52_STREAM) but does not get
        // 2 MiB pages: a 147 GiB tier on 4 KiB pages is ~39M page-table entries
        // and scattered DMA out of it measured 14% slower prefill (99.6 -> 86.0
        // tok/s, run13). The anonymous arena keeps MADV_HUGEPAGE and is what
        // every copy-path number in this log was measured on, so it stays the
        // default; malloc_host is used only when streaming asks for it.
        if (g_stream) {
            hot_base = reinterpret_cast<uint8_t*>(1);
            hot_ptr.assign(size_t(NLA)*NE, nullptr);
            std::printf("hot tier: up to %.1f GiB of malloc_host USM (kernel-readable, "
                        "4 KiB pages)\n", hot_cap/1073741824.0);
        } else
        if (posix_memalign((void**)&hot_base, 2u<<20, hot_cap) != 0) hot_base = nullptr;
        if (hot_base && !g_stream) {
            if (madvise(hot_base, hot_cap, MADV_HUGEPAGE) != 0)
                std::printf("  (MADV_HUGEPAGE refused; tier stays on 4 KiB pages)\n");
            {   // Pre-fault the whole tier NOW, while physical memory is still
                // unfragmented. Touched incrementally over a ~30-minute fill
                // while the page cache churns, THP coverage silently collapses
                // to 4 KiB pages and expert DMA halves (measured 26.3 ->
                // 14.1 GB/s across three consecutive loads). One memset claims
                // 2 MiB pages up front — measured 100% coverage at 11.6 GB/s.
                const auto t_pf = Clock::now();
                std::memset(hot_base, 0, hot_cap);
                unsigned long long thp_kb = 0;
                if (FILE* sm = fopen("/proc/self/smaps_rollup", "r")) {
                    char ln[256];
                    while (fgets(ln, sizeof ln, sm))
                        if (sscanf(ln, "AnonHugePages: %llu", &thp_kb) == 1) break;
                    fclose(sm);
                }
                std::printf("  tier pre-faulted in %.1f s; AnonHugePages %.2f GiB\n",
                            dsec(t_pf), double(thp_kb)/1048576.0);
            }
            hot_ptr.assign(size_t(NLA)*NE, nullptr);
            std::printf("hot tier: %.1f GiB anonymous, registered with %d context(s)\n",
                        hot_cap/1073741824.0, NG);
        }
    }

    std::vector<std::vector<std::pair<int,int64_t>>> static_map{static_cast<size_t>(NG)};
    std::vector<Cache> CA;
    for (int d = 0; d < NG; ++d) {
        // Size each cache from what is ACTUALLY free on that card, not from a
        // fixed slot count. GPU0 already holds 13.96 GiB of resident weights plus
        // the display; taking a fixed 14.5 GiB cache on top of that overcommitted
        // it and the run died at first kernel launch with
        // UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY. Headroom covers the framebuffer,
        // the KV cache, working buffers and driver overhead.
        const uint64_t total_vram = devs[d].get_info<sycl::info::device::global_mem_size>();
        // Headroom must cover the KV cache (0.35 GiB at 2108 ctx) and the T*
        // prefill tensors (~15 x TB*H*4 = ~0.8 GiB) — about 1.2 GiB in total.
        // 4 GiB was chosen before those were measured. Every GiB returned is
        // ~104 more expert slots per card, which cuts prefill's residency phase
        // (7.18 s of 21.4) and decode's fold (gpu1's fetch) at the same time.
        // IE_GLM52_HEADROOM_MB overrides.
        const uint64_t headroom   = getenv("IE_GLM52_HEADROOM_MB")
                                    ? (uint64_t(atoll(getenv("IE_GLM52_HEADROOM_MB"))) << 20)
                                    : (2048ull << 20);
        const uint64_t taken      = (d == 0) ? vram_used : vram_used1;
        const uint64_t avail      = (total_vram > taken + headroom)
                                    ? total_vram - taken - headroom : 0;
        int n = int(avail / slot_bytes);
        if (cache_experts > 0 && n > cache_experts) n = cache_experts;  // user cap
        CA.push_back(Cache{ sycl::queue(ctxs[d], devs[d], qprops()),
                            sycl::queue(ctxs[d], devs[d], qprops()),
                            {}, {}, {}, {} });
        auto& c = CA.back();
        c.slot.resize(n); c.key.assign(n,-1); c.age.assign(n,0); c.fill.resize(n);
        for (int i = 0; i < 4; ++i)
            c.qxs.emplace_back(ctxs[d], devs[d], qprops());
        c.lookup.assign(size_t(NLA)*NE, -1);
        // Allocate the slots in ARENAS, not one malloc_device each. The MoE
        // kernels take an array of device pointers and dereference it (indirect
        // USM access), and the Level-Zero adapter makes every known allocation
        // resident per such launch — so submission cost scales with the
        // allocation COUNT. 2,800 slot allocations per card is the difference
        // between the microbenchmark's 9 us launch and the ~1 ms/block the
        // engine measures. IE_GLM52_ARENA sets slots per arena (0 = one per).
        // MEASURED: submission cost is flat (2.1-4.3 us) from 8 to 4,000 live
        // allocations, so allocation count is NOT the per-submission tax.
        // Arenas stay available but default OFF.
        // Arena allocation is now ON by default. Growing the cache 4,120 -> 4,418
        // INDIVIDUAL slot allocations cost 38% of decode (8.17 -> 5.09 tok/s)
        // even though the hit rate improved and bytes fell; the same cache behind
        // 36 arenas recovers it (7.81) and drops the residency loop to 0.76
        // ms/token. Per-miss cost evidently scales with the number of distinct
        // device allocations cycled through, which the earlier "8 vs 4,000
        // allocations" microbenchmark missed because it reused the same buffers.
        // Measured decode2, one run each:
        //   3,824 slots + arena      7.30 tok/s
        //   4,120 slots, no arena    8.17 and 7.64  (same config, twice)
        //   4,418 slots + arena      7.81
        //   4,418 slots, no arena    5.09   <-- a real cliff
        // The identical config measured 8.17 and 7.64, so run-to-run spread is
        // ~6.5% and every size above is one band. Only the cliff is real: past
        // ~4,100 INDIVIDUAL slot allocations, per-miss cost jumps. Arena removes
        // it and costs nothing measurable, so it is on by default — the robust
        // choice rather than whichever size won a single noisy comparison.
        const int arena = getenv("IE_GLM52_ARENA") ? atoi(getenv("IE_GLM52_ARENA")) : 128;
        if (arena <= 1) {
            for (int i = 0; i < n; ++i) {
                c.slot[i] = sycl::malloc_device<uint8_t>(slot_bytes, devs[d], ctxs[d]);
                if (!c.slot[i]) { c.slot.resize(i); c.key.resize(i); c.age.resize(i); break; }
            }
        } else {
            int done = 0;
            while (done < n) {
                const int take = std::min(arena, n - done);
                uint8_t* base = sycl::malloc_device<uint8_t>(size_t(take)*slot_bytes,
                                                            devs[d], ctxs[d]);
                if (!base) {
                    if (take <= 1) break;
                    // shrink the request rather than give up the whole tail
                    uint8_t* half = sycl::malloc_device<uint8_t>(size_t(take/2)*slot_bytes,
                                                                devs[d], ctxs[d]);
                    if (!half) break;
                    for (int i = 0; i < take/2; ++i) c.slot[done+i] = half + size_t(i)*slot_bytes;
                    done += take/2;
                    continue;
                }
                for (int i = 0; i < take; ++i) c.slot[done+i] = base + size_t(i)*slot_bytes;
                done += take;
            }
            if (done < n) { c.slot.resize(done); c.key.resize(done); c.age.resize(done);
                            c.fill.resize(done); }
            std::printf("gpu%d slot arenas: %d slots in %d allocation(s)\n",
                        d, done, (done + arena - 1)/arena);
        }
        // GPU0's cache queues must BE the main queues, not merely queues on the
        // same device: the expert kernels write EWs[0].moe and the shared-expert
        // add reads it from `q`. Two distinct in-order queues give no ordering
        // between them, so this would be a silent read-before-write race.
        if (d == 0) { c.qc = q; c.qx = qt; }
        if (hot_base && !g_stream)
            sycl::ext::oneapi::experimental::prepare_for_device_copy(hot_base, hot_cap, ctxs[d]);
        c.stage.resize(STAGE_RING);
        c.stage_ev.resize(STAGE_RING);
        for (int i = 0; i < STAGE_RING; ++i) {
            c.stage[i] = sycl::malloc_host<uint8_t>(slot_bytes, ctxs[d]);
            if (!c.stage[i]) { std::fprintf(stderr,"staging buffer alloc failed\n"); return 1; }
        }
        std::printf("gpu%d expert cache: %zu slots x %.2f MiB = %.2f GiB\n",
                    d, c.slot.size(), slot_bytes/1048576.0,
                    double(c.slot.size())*slot_bytes/1073741824.0);
    }
    std::printf("\n");
    uint64_t clk = 0;

    // ---- preload: hottest owned experts, permanently resident in VRAM ------
    {
        auto t_pre = Clock::now();
        std::vector<std::pair<int32_t,size_t>> byrank;   // (rank, key)
        for (size_t k = 0; k < rank.size(); ++k)
            if (rank[k] != INT32_MAX) byrank.push_back({rank[k], k});
        std::sort(byrank.begin(), byrank.end());

        // (slot, key) of every statically placed expert. A config that releases
        // static slots to decode's LRU lets them be overwritten, so the next
        // config would not start from the same cache — restoring them from the
        // host tier costs ~1.3 s and keeps configs comparable.
        std::vector<size_t> filled(NG, 0);
        for (int d = 0; d < NG; ++d) {
            CA[d].static_cap = (CA[d].slot.size() * size_t(g_static_pct)) / 100;
            std::printf("gpu%d static cap %zu of %zu slots (%zu reserved for demand)\n",
                        d, CA[d].static_cap, CA[d].slot.size(),
                        CA[d].slot.size() - CA[d].static_cap);
        }
        std::vector<uint8_t> buf(slot_bytes);
        uint64_t pre_bytes = 0;
        for (auto& [rk, key] : byrank) {
            const int d = owner[key];
            Cache& c = CA[d];
            // Leave a reservoir of unpinned slots. Filling EVERY slot with a
            // pinned resident makes the LRU victim search compare equal ages,
            // pick slot 0, and evict a hot resident on every miss — which
            // thrashes far worse than having no static placement at all.
            if (filled[d] >= c.static_cap) continue;
            const int il = int(key / NE), e = int(key % NE);
            const Layer& l = L[il];
            // pread straight into the staging path, then park it in a slot that
            // LRU can never evict (age = UINT64_MAX).
            std::vector<PreadPool::Job> jb = {
                {gfd, buf.data(),                l.off_gate + uint64_t(e)*gu_bytes, gu_bytes},
                {gfd, buf.data() + gu_bytes,     l.off_up   + uint64_t(e)*gu_bytes, gu_bytes},
                {gfd, buf.data() + 2*gu_bytes,   l.off_down + uint64_t(e)*dn_bytes, dn_bytes}};
            pool.run(jb);
            const size_t sidx = filled[d]++;
            c.qx.memcpy(c.slot[sidx], buf.data(), slot_bytes).wait();
            c.key[sidx] = int64_t(key);
            c.lookup[key] = int(sidx);
            c.age[sidx] = UINT64_MAX;                      // pinned: never evicted
            static_map[size_t(d)].push_back({int(sidx), int64_t(key)});
            pre_bytes += slot_bytes;
            if ((pre_bytes >> 30) && ((pre_bytes / slot_bytes) % 512 == 0))
                std::printf("  preloaded %.1f GiB (%.0f s)\n",
                            pre_bytes/1073741824.0, dsec(t_pre));
        }
        // Only reserve what was actually placed. Holding 7/8 of the cache for a
        // static set that never arrived (no calibration trace) left 186 demand
        // slots against 256 experts per block and thrashed to a 0% hit rate.
        for (int d = 0; d < NG; ++d) {
            if (filled[d] < CA[d].static_cap) {
                std::printf("gpu%d static cap %zu -> %zu (rest released to demand)\n",
                            d, CA[d].static_cap, filled[d]);
                CA[d].static_cap = filled[d];
            }
        }
        for (int d = 0; d < NG; ++d)
            std::printf("gpu%d resident experts: %zu / %zu slots\n",
                        d, filled[d], CA[d].slot.size());
        std::printf("VRAM preload: %.2f GiB in %.0f s\n", pre_bytes/1073741824.0, dsec(t_pre));

        // ---- fill the host tier with everything else --------------------
        // The whole point of 256 GB of RAM: the 182 GiB expert bank lives there,
        // registered, at 26 GB/s. Anything not resident in VRAM is then ONE DMA
        // away instead of a disk seek. Filling on demand instead leaves the first
        // pass reading a 120 MB/s spinning disk, which is what made the previous
        // run collapse to 0.15 tok/s.
        if (hot_base) {
            auto t_tier = Clock::now();
            std::vector<uint8_t> tb(slot_bytes);
            uint64_t n_tier = 0;
            auto push = [&](int il, int e) {
                const size_t key = size_t(il)*NE + e;
                if (hot_ptr[key]) return;                       // already there
                if (CA[owner[key]].lookup[key] >= 0) return;    // resident in VRAM
                if (hot_used + slot_bytes > hot_cap) return;
                uint8_t* hot = hot_alloc(slot_bytes);
                if (!hot) return;
                const Layer& l2 = L[il];
                std::vector<PreadPool::Job> jb = {
                    {gfd, tb.data(),              l2.off_gate + uint64_t(e)*gu_bytes, gu_bytes},
                    {gfd, tb.data() + gu_bytes,   l2.off_up   + uint64_t(e)*gu_bytes, gu_bytes},
                    {gfd, tb.data() + 2*gu_bytes, l2.off_down + uint64_t(e)*dn_bytes, dn_bytes}};
                pool.run(jb);
                std::memcpy(hot, tb.data(), slot_bytes);
                hot_ptr[key] = hot;
                ++n_tier;
                posix_fadvise(gfd, off_t(l2.off_gate + uint64_t(e)*gu_bytes), off_t(gu_bytes), POSIX_FADV_DONTNEED);
                posix_fadvise(gfd, off_t(l2.off_up   + uint64_t(e)*gu_bytes), off_t(gu_bytes), POSIX_FADV_DONTNEED);
                posix_fadvise(gfd, off_t(l2.off_down + uint64_t(e)*dn_bytes), off_t(dn_bytes), POSIX_FADV_DONTNEED);
                if ((n_tier % 1024) == 0)
                    std::printf("  tier %.1f GiB (%.0f s, %.0f MB/s)\n",
                                hot_used/1073741824.0, dsec(t_tier),
                                hot_used/dsec(t_tier)/1e6);
            };
            // Preload the ranked working set only. Filling the cold tail costs
            // ~20 min of 86 MB/s disk for experts this workload never touches;
            // leaving that capacity free lets demand-promotion absorb whatever
            // the calibration trace missed (it was taken over 36 tokens, and a
            // longer generation visits experts it never saw).
            for (auto& [rk, key] : byrank) push(int(key / NE), int(key % NE));
            if (getenv("IE_GLM52_TIER_ALL"))
                // NLA includes the MTP block's expert bank when speculative
                // decode is on — leaving it out sent every MTP draft's misses
                // to a 120 MB/s disk pread.
                for (int il = 0; il < NLA; ++il) {
                    // Blocks 0-2 are DENSE: they have no expert bank at all, and
                    // pushing them read garbage from the file and burned 7.2 GiB
                    // of the tier. That pushed ~500 real experts off the end onto
                    // the spinning disk -- "hot tier served 98.7% of misses", and
                    // the 1.3% that missed cost ~7 s of a 47 s prefill, because
                    // each one is a blocking 9.6 MiB read at 120 MB/s.
                    if (C.is_dense_layer(uint32_t(il))) continue;
                    for (int e = 0; e < NE; ++e) push(il, e);
                }
            std::printf("host tier: %.2f GiB, %llu experts, in %.0f s\n\n",
                        hot_used/1073741824.0, (unsigned long long)n_tier, dsec(t_tier));
        }
    }

    // ---- working buffers ---------------------------------------------------
    auto dalloc = [&](size_t n){ return sycl::malloc_device<float>(n, dev, ctx); };
    float* x       = dalloc(H);
    float* xb      = dalloc(H);
    float* qa      = dalloc(QL);
    float* qb      = dalloc(size_t(NH)*C.key_len_mla);
    float* kv      = dalloc(KVL + RL);
    float* ckv_n   = dalloc(KVL);
    float* qc      = dalloc(size_t(NH)*KVL);
    float* qr      = dalloc(size_t(NH)*RL);
    float* oc      = dalloc(size_t(NH)*KVL);
    float* ov      = dalloc(size_t(NH)*VHD);
    float* attn_o  = dalloc(H);
    float* ffn_g   = dalloc(std::max<size_t>(C.ffn, EF));
    float* ffn_u   = dalloc(std::max<size_t>(C.ffn, EF));
    float* ffn_a   = dalloc(std::max<size_t>(C.ffn, EF));
    float* rlogits = dalloc(NE);
    float* logits  = dalloc(VOC);
    float* scratch = dalloc(size_t(NH)*n_ctx);
    float* kcache  = dalloc(size_t(NL)*n_ctx*KVL);
    float* rcache  = dalloc(size_t(NL)*n_ctx*RL);
    // ---- batched-prefill buffers ------------------------------------------
    // Prefill processes the whole prompt in ONE pass so the 13.4 GiB resident
    // set is read once per BATCH instead of once per token, and each expert's
    // trellis weights are decoded once for every token routed to it.
    // Prefill runs in chunks. The attention scratch is T x heads x ctx floats,
    // so an unchunked 512-token prompt would want ~70 GB; chunking bounds every
    // batch buffer while still amortising weights across the chunk.
    const int CH = getenv("IE_GLM52_CHUNK") ? atoi(getenv("IE_GLM52_CHUNK")) : 64;
    const int TB = std::max(1, std::min<int>(CH, std::max<int>(1, int(prompt.size()))));
    const bool use_xmx = getenv("IE_GLM52_XMX") != nullptr;
    if (const char* e = getenv("IE_GLM52_GEMM")) g_gemm_v2 = atoi(e);

    std::printf("prefill chunk: %d tokens%s\n", TB, use_xmx ? "  [XMX expert GEMM]" : "");
    float* Txb   = dalloc(size_t(TB)*H);
    float* Tqa   = dalloc(size_t(TB)*QL);
    float* Tqb   = dalloc(size_t(TB)*NH*C.key_len_mla);
    float* Tkv   = dalloc(size_t(TB)*(KVL+RL));
    float* Tckv  = dalloc(size_t(TB)*KVL);
    float* Tkr   = dalloc(size_t(TB)*RL);
    float* Tqc   = dalloc(size_t(TB)*NH*KVL);
    float* Tqr   = dalloc(size_t(TB)*NH*RL);
    float* Toc   = dalloc(size_t(TB)*NH*KVL);
    float* Tov   = dalloc(size_t(TB)*NH*VHD);
    float* Tattn = dalloc(size_t(TB)*H);
    float* Tx    = dalloc(size_t(TB)*H);
    float* Tffg  = dalloc(size_t(TB)*std::max<size_t>(C.ffn, EF));
    float* Tffu  = dalloc(size_t(TB)*std::max<size_t>(C.ffn, EF));
    float* Tffa  = dalloc(size_t(TB)*std::max<size_t>(C.ffn, EF));
    float* Trl   = dalloc(size_t(TB)*NE);
    float* Tmoe  = dalloc(size_t(TB)*H);
    // (score scratch retired — k_mla_flash_T streams the softmax)
    float* Tgu   = dalloc(size_t(TB)*EF);
    float* Tup   = dalloc(size_t(TB)*EF);
    float* Tdn   = dalloc(size_t(TB)*H);
    int32_t* Ttok = sycl::malloc_shared<int32_t>(size_t(TB), devs[0], ctxs[0]);
    float*   Twt  = sycl::malloc_shared<float>(size_t(TB), devs[0], ctxs[0]);
    int32_t* Tseq = sycl::malloc_device<int32_t>(size_t(TB), devs[0], ctxs[0]);
    { std::vector<int32_t> hs(static_cast<size_t>(TB), 0); for (int i = 0; i < TB; ++i) hs[i] = i;
      q.memcpy(Tseq, hs.data(), size_t(TB)*4).wait(); }
    // All of a block's expert token-lists laid out contiguously (there are
    // exactly TOPK*T selections per block). One buffer per block instead of one
    // reused buffer per expert removes a pipeline stall per expert — that was
    // ~5,250 syncs per batch and it dominated prefill.
    // DEVICE memory, not shared USM. These are read per work-item by the expert
    // GEMM and the scatter-add (nt*H = ~98k items), and shared-USM reads in a hot
    // kernel fault a page at a time — it made a 512-token prefill take >25 min.
    float*   Tbias = sycl::malloc_device<float>(size_t(NE), devs[0], ctxs[0]);
    int32_t* Ttopi = sycl::malloc_device<int32_t>(size_t(TB)*TOPK, devs[0], ctxs[0]);
    float*   Ttopw = sycl::malloc_device<float>(size_t(TB)*TOPK, devs[0], ctxs[0]);
    std::vector<int32_t> Htopi(static_cast<size_t>(TB)*TOPK, 0);
    std::vector<float>   Htopw(static_cast<size_t>(TB)*TOPK, 0.f);
    int32_t* Tallt = sycl::malloc_device<int32_t>(size_t(TB)*TOPK, devs[0], ctxs[0]);
    float*   Tallw = sycl::malloc_device<float>(size_t(TB)*TOPK, devs[0], ctxs[0]);
    std::vector<int32_t> Hallt(static_cast<size_t>(TB)*TOPK, 0);
    std::vector<float>   Hallw(static_cast<size_t>(TB)*TOPK, 0.f);
    // g_rp: second buffer + last-H2D events, so the pack of block il waits only
    // on block il-2's copies instead of draining both compute queues per block.
    std::vector<int32_t> HalltB(static_cast<size_t>(TB)*TOPK, 0);
    std::vector<float>   HallwB(static_cast<size_t>(TB)*TOPK, 0.f);
    sycl::event hall_evq[2], hall_ev1[2];
    float*   Tgub  = dalloc(size_t(TB)*TOPK*EF);
    float*   Tupb  = dalloc(size_t(TB)*TOPK*EF);
    float*   Tdnb  = dalloc(size_t(TB)*TOPK*H);
    // Logits for every row of the SPECULATIVE VERIFY WINDOW -- never a whole
    // prefill chunk, so this is sized by the window, not TB. At a 2048-token
    // chunk the difference is 1.27 GiB of VRAM that the expert cache gets back.
    const int TLOGR = std::min(TB, 72);
    float*   Tlog  = dalloc(size_t(TLOGR)*VOC);
    // GPU1 mirrors of the batched MoE working set, so the batched path (prefill
    // AND speculative verify) uses both cards the way decode does. Running it on
    // gpu0 alone made verify slower than plain sequential decode.
    float* T1xb  = (NG>1) ? sycl::malloc_device<float>(size_t(TB)*H, devs[1], ctxs[1]) : nullptr;
    float* T1moe = (NG>1) ? sycl::malloc_device<float>(size_t(TB)*H, devs[1], ctxs[1]) : nullptr;
    // Zero-copy fold: the grouped path's gather on gpu1 writes its partial
    // straight into host memory (one streamed 50 MB PCIe write inside the
    // kernel), which deletes the 2.5 ms/block device-to-host leg of the fold.
    // Registered with gpu0's context so the H2D into Tfold runs at link rate.
    float* T1moeH = (NG>1) ? sycl::malloc_host<float>(size_t(TB)*H, ctxs[1]) : nullptr;
    if (T1moeH)
        sycl::ext::oneapi::experimental::prepare_for_device_copy(
            T1moeH, size_t(TB)*size_t(H)*4, ctxs[0]);
    float* T1gu  = (NG>1) ? sycl::malloc_device<float>(size_t(TB)*TOPK*EF, devs[1], ctxs[1]) : nullptr;
    float* T1up  = (NG>1) ? sycl::malloc_device<float>(size_t(TB)*TOPK*EF, devs[1], ctxs[1]) : nullptr;
    float* T1dn  = (NG>1) ? sycl::malloc_device<float>(size_t(TB)*TOPK*H, devs[1], ctxs[1]) : nullptr;
    int32_t* T1t = (NG>1) ? sycl::malloc_device<int32_t>(size_t(TB)*TOPK, devs[1], ctxs[1]) : nullptr;
    float*   T1w = (NG>1) ? sycl::malloc_device<float>(size_t(TB)*TOPK, devs[1], ctxs[1]) : nullptr;
    int32_t* T1seq = (NG>1) ? sycl::malloc_device<int32_t>(size_t(TB), devs[1], ctxs[1]) : nullptr;
    // ---- tensor-parallel attention buffers on GPU1 -------------------------
    const int NHT = want_tp ? NH/2 : NH;          // heads per card in prefill
    auto d1alloc = [&](size_t n){ return want_tp ? sycl::malloc_device<float>(n, devs[1], ctxs[1])
                                                 : nullptr; };
    float* T1qa  = d1alloc(size_t(TB)*QL);
    float* T1qb  = d1alloc(size_t(TB)*NHT*C.key_len_mla);
    float* T1qc  = d1alloc(size_t(TB)*NHT*KVL);
    float* T1qr  = d1alloc(size_t(TB)*NHT*RL);
    float* T1oc  = d1alloc(size_t(TB)*NHT*KVL);
    float* T1ov  = d1alloc(size_t(TB)*NHT*VHD);
    float* T1att = d1alloc(size_t(TB)*H);
    float* T1ffg = d1alloc(size_t(TB)*(EF/2));
    float* T1ffu = d1alloc(size_t(TB)*(EF/2));
    float* T1ckv = d1alloc(size_t(TB)*KVL);
    float* T1kr  = d1alloc(size_t(TB)*RL);
    // GPU1 needs the whole latent KV cache: MLA shares it across all heads.
    float* kcache1 = d1alloc(size_t(NL)*n_ctx*KVL);
    float* rcache1 = d1alloc(size_t(NL)*n_ctx*RL);
    float* Tov1d = want_tp ? dalloc(size_t(TB)*NHT*VHD) : nullptr;   // gpu1's half, landed on gpu0
    std::vector<float> Hqa (want_tp ? size_t(TB)*QL : 1, 0.f);
    std::vector<float> Hkv (want_tp ? size_t(TB)*(KVL+RL) : 1, 0.f);
    std::vector<float> Hov (want_tp ? size_t(TB)*NHT*VHD : 1, 0.f);
    std::vector<float> Tbounce(static_cast<size_t>(TB)*H, 0.f);
    float* Tfold = (NG > 1) ? dalloc(size_t(TB)*H) : nullptr;
    // Per-card expert descriptors for the single-launch MoE: base pointers,
    // token-slice offset/count, and the output row offset.
    // Per-card MoE descriptors, ROUND-ROBIN over MDSLOT staging buffers.
    //
    // These used to be seven separate device arrays copied with seven memcpys
    // followed by `qq.wait()` -- because the source std::vectors were reused
    // immediately. That wait drains the whole compute queue, so it also
    // destroyed any chance of overlapping one group's expert DMA with the
    // previous group's GEMMs. Staging in pinned host memory across several slots
    // removes the wait entirely: one H2D per launch, and a slot is not rewritten
    // until MDSLOT launches later, by which point its copy has long completed.
    constexpr int MDSLOT = 8;
    const size_t md_tiles = size_t(TB)*TOPK + 2*size_t(NE);   // (expert, m0) pairs
    struct MoEDesc {
        uint8_t* hstage[MDSLOT]; uint8_t* dstage[MDSLOT];
        sycl::event ev[MDSLOT];
        // Descriptors go on their OWN queue. They used to ride the compute
        // queue, which meant the host wait before reusing a staging slot waited
        // for all prior GEMMs -- stalling the host exactly when it should have
        // been issuing the next group's expert DMA. A direct microbenchmark says
        // copies and kernels overlap perfectly on this hardware, so the missing
        // overlap was this, not the platform.
        std::vector<sycl::queue> qd;
        size_t o_base, o_eoff, o_ecnt, o_goff, o_tile, o_rtok, o_rw, bytes;
        int slot = 0;
    };
    std::vector<MoEDesc> MD{static_cast<size_t>(NG)};
    for (int d = 0; d < NG; ++d) {
        MoEDesc& M = MD[size_t(d)];
        size_t o = 0;
        M.o_base = o; o += size_t(NE)*sizeof(uint8_t*);
        M.o_eoff = o; o += size_t(NE)*4;
        M.o_ecnt = o; o += size_t(NE)*4;
        M.o_goff = o; o += size_t(NE)*4;
        M.o_tile = o; o += md_tiles*2*4;
        M.o_rtok = o; o += size_t(TB)*TOPK*4;
        M.o_rw   = o; o += size_t(TB)*TOPK*4;
        M.bytes  = o;
        for (int k = 0; k < MDSLOT; ++k) {
            M.hstage[k] = static_cast<uint8_t*>(sycl::malloc_host(M.bytes, ctxs[d]));
            M.dstage[k] = sycl::malloc_device<uint8_t>(M.bytes, devs[d], ctxs[d]);
        }
        M.qd.emplace_back(ctxs[d], devs[d], qprops());
    }
    if (NG > 1) { std::vector<int32_t> hs(static_cast<size_t>(TB), 0);
                  for (int i = 0; i < TB; ++i) hs[i] = i;
                  CA[1].qc.memcpy(T1seq, hs.data(), size_t(TB)*4).wait(); }

    std::vector<std::pair<float,int>> sel(NE);
    // Per-GPU expert working set. GPU0 reuses the main xb; GPU1 gets a copy of
    // the post-ffn_norm activation and returns a partial MoE output.
    struct EW {
        float *xb, *gu, *out, *moe, *wts;
        uint8_t **bases, **ub, **db;
        int lo, hi;                        // (unused once routing is by ownership)
        int nk;                            // experts this card owns for this block
        // decode v2: hit/miss split scratch (b2/u2/d2/w2 compacted hits-then-
        // misses, smap maps compact index -> canonical slot), per-expert down
        // rows, and per-slot fill events + miss flags.
        // Staged descriptor blob: [bases|ub|db][wts], in a RING. The copy is
        // async and the host rewrites the staging buffer on the very next block,
        // so a single buffer is a write-after-read race — the kernel then
        // dereferences half-updated pointers and the GPU hangs. Prefill uses the
        // same ring for the same reason (MDSLOT).
        uint8_t **hb_bases, **hb_ub, **hb_db; float* hb_wts;
        uint8_t *dblob[8], *hblob[8]; sycl::event dev_ev[8];
        int dslot = 0; size_t blob_bytes;
        float *dnrows, *w2; int32_t *smap;
        uint8_t **b2, **u2, **d2;
        std::array<sycl::event,16> fev;     // gates gate/up (miss: gate+up copy)
        std::array<sycl::event,16> fev_dn;  // gates down    (miss: down copy)
        std::array<uint8_t,16>     mflag;
    };
    std::vector<EW> EWs(NG);
    for (int d = 0; d < NG; ++d) {
        EW& e = EWs[d];
        e.lo = 0; e.hi = TOPK;               // set per run below
        const int nk = TOPK;                  // size for the worst case
        e.xb    = (d==0) ? xb : sycl::malloc_device<float>(size_t(H), devs[d], ctxs[d]);
        e.gu    = sycl::malloc_device<float>(size_t(EF)*nk, devs[d], ctxs[d]);
        e.out   = sycl::malloc_device<float>(std::max<size_t>(size_t(EF)*nk, H), devs[d], ctxs[d]);
        e.moe   = sycl::malloc_device<float>(size_t(H), devs[d], ctxs[d]);
        e.wts   = sycl::malloc_shared<float>(size_t(nk), devs[d], ctxs[d]);
        e.bases = sycl::malloc_shared<uint8_t*>(size_t(nk), devs[d], ctxs[d]);
        e.ub    = sycl::malloc_shared<uint8_t*>(size_t(nk), devs[d], ctxs[d]);
        e.db    = sycl::malloc_shared<uint8_t*>(size_t(nk), devs[d], ctxs[d]);
        e.dnrows= sycl::malloc_device<float>(size_t(H)*nk, devs[d], ctxs[d]);
        e.smap  = sycl::malloc_shared<int32_t>(size_t(nk), devs[d], ctxs[d]);
        e.b2    = sycl::malloc_shared<uint8_t*>(size_t(nk), devs[d], ctxs[d]);
        e.u2    = sycl::malloc_shared<uint8_t*>(size_t(nk), devs[d], ctxs[d]);
        e.d2    = sycl::malloc_shared<uint8_t*>(size_t(nk), devs[d], ctxs[d]);
        e.w2    = sycl::malloc_shared<float>(size_t(nk), devs[d], ctxs[d]);
        // Host-side staging for the descriptors, plus one device blob they are
        // copied into once per block. Host memory the host writes, device memory
        // the device reads — no page ping-pong.
        e.blob_bytes = size_t(nk)*3*sizeof(uint8_t*) + size_t(nk)*sizeof(float);
        for (int q2 = 0; q2 < 8; ++q2) {
            e.hblob[q2] = static_cast<uint8_t*>(sycl::malloc_host(e.blob_bytes, ctxs[d]));
            e.dblob[q2] = sycl::malloc_device<uint8_t>(e.blob_bytes, devs[d], ctxs[d]);
        }
        e.dslot = 0;
        e.hb_bases = reinterpret_cast<uint8_t**>(e.hblob[0]);
        e.hb_ub    = e.hb_bases + nk;
        e.hb_db    = e.hb_ub + nk;
        e.hb_wts   = reinterpret_cast<float*>(e.hb_db + nk);
    }
    float* moe_partial = (NG > 1) ? sycl::malloc_device<float>(size_t(H), devs[0], ctxs[0]) : nullptr;
    // Plain host memory: usable from either context, unlike USM tied to one.
    std::vector<float> bounce(size_t(H), 0.f);
    // How many GPUs the CURRENT generation uses. Both configurations are measured
    // from ONE load, because populating the pinned bank costs ~27 minutes off the
    // Data1 HDD and paying that twice to A/B one variable is not worth it.
    int ng_use = NG;
    vram_used += (size_t(NL)*n_ctx*(KVL+RL))*4;
    std::printf("KV cache: %.2f GiB for %d ctx\n\n",
                double(size_t(NL)*n_ctx*(KVL+RL))*4/1073741824.0, n_ctx);

    const bool PROF = getenv("IE_GLM52_PROFILE") != nullptr;
    // IE_GLM52_TRACE=<path> dumps the router's expert choices per token per block.
    FILE* trace = getenv("IE_GLM52_TRACE") ? fopen(getenv("IE_GLM52_TRACE"), "w") : nullptr;
    double t_attn=0, t_router=0, t_exp=0, t_shexp=0, t_head=0, t_fetch=0;
    auto phase = [&](double& acc, Clock::time_point& t){
        if (!PROF) return; q.wait(); acc += dsec(t); t = Clock::now(); };
    // Prefill has its own breakdown: the single-token counters above never see
    // forward_chunk, which is where prompt-processing time actually goes.
    double p_embed=0, p_attn=0, p_route=0, p_moe=0, p_dense=0, p_head=0;
    double p_fetch=0;
    auto pphase = [&](double& acc, Clock::time_point& t){
        if (!PROF) return;
        q.wait(); if (NG > 1) CA[1].qc.wait();
        acc += dsec(t); t = Clock::now(); };

    float* wo_part = want_tp ? dalloc(H) : nullptr;   // second wo K-slice partial
    // Last token's expert choice per block, used to prefetch the next block's.
    std::vector<std::array<int,16>> prev_sel(static_cast<size_t>(NL));
    for (auto& a : prev_sel) a.fill(-1);
    // Predictive prefetch from the previous token's routing: measured useless
    // (287 transfers issued over 7,600 block-visits, decode unchanged), because
    // anything it predicts is already resident. Off unless asked for.
    bool do_prefetch = getenv("IE_GLM52_PREFETCH") != nullptr;
    size_t g_qx = 1;   // transfer queues per card; 2 and 4 measured no different
    std::vector<float> h_rlogits(NE), h_probs(NE);
    std::vector<float> h_embed(H);

    // ---- MTP buffers -------------------------------------------------------
    float* mtp_e   = want_mtp ? dalloc(H) : nullptr;
    float* mtp_cat = want_mtp ? dalloc(size_t(2)*H) : nullptr;
    float* mtp_cur = want_mtp ? dalloc(H) : nullptr;
    float* mtp_sa  = want_mtp ? dalloc(H) : nullptr;
    float* mtp_kc  = want_mtp ? dalloc(size_t(n_ctx)*KVL) : nullptr;
    float* mtp_rc  = want_mtp ? dalloc(size_t(n_ctx)*RL) : nullptr;
    std::vector<float> mtp_hemb(static_cast<size_t>(H), 0.f);

    // One NextN block: draft the token that follows `next_tok`, given the main
    // model's hidden state for the step that produced it. Graph per
    // ik_llama build_deepseek2_mtp / build_mtp_input.
    auto mtp_step = [&](const float* hidden, int next_tok, int pos) {
        if (!want_mtp) return;
        Layer& l = L[NL];
        {   // embed(next_tok)
            const auto* ti = g.find_tensor("token_embd.weight");
            const int64_t src_row = (int64_t(H)/32)*26;
            const uint8_t* rp = ti->data + int64_t(next_tok)*src_row;
            for (int b = 0; b < H/32; ++b) {
                const uint8_t* blk = rp + b*26;
                sycl::half dh; std::memcpy(&dh, blk, 2);
                const float d = float(dh);
                for (int j = 0; j < 16; ++j) {
                    const int l0 = blk[10+j] & 0x0F, l1 = blk[10+j] >> 4;
                    const int h0 = (blk[2 + (j%8)] >> (4*(j/8))) & 3;
                    const int h1 = (blk[2 + (j%8)] >> (4*(j/8)+2)) & 3;
                    mtp_hemb[b*32 + j]      = d*float((l0 | (h0<<4)) - 32);
                    mtp_hemb[b*32 + j + 16] = d*float((l1 | (h1<<4)) - 32);
                }
            }
            q.memcpy(mtp_e, mtp_hemb.data(), size_t(H)*4).wait();
        }
        // concat(RMSNorm(e, enorm), RMSNorm(hidden, hnorm)) -> eh_proj
        k_rmsnorm(q, mtp_e, mtp_enorm, mtp_cat,     H, C.rms_eps);
        k_rmsnorm(q, hidden, mtp_hnorm, mtp_cat + H, H, C.rms_eps);
        k_q6_gemv(q, mtp_eh.dev, mtp_cat, mtp_cur, mtp_eh.K, mtp_eh.N, mtp_eh.row_bytes);
        q.memcpy(mtp_sa, mtp_cur, size_t(H)*4);

        // ---- MLA on the MTP block, own KV cache ----
        k_rmsnorm(q, mtp_cur, l.attn_norm, xb, H, C.rms_eps);
        k_q6_gemv(q, l.wq_a.dev, xb, qa, l.wq_a.K, l.wq_a.N, l.wq_a.row_bytes);
        k_rmsnorm(q, qa, l.q_a_norm, qa, QL, C.rms_eps);
        k_q6_gemv(q, l.wq_b.dev, qa, qb, l.wq_b.K, l.wq_b.N, l.wq_b.row_bytes);
        k_q6_gemv(q, l.wkv_a.dev, xb, kv, l.wkv_a.K, l.wkv_a.N, l.wkv_a.row_bytes);
        k_rmsnorm(q, kv, l.kv_a_norm, ckv_n, KVL, C.rms_eps);
        {
            float* kc = mtp_kc + size_t(pos)*KVL;
            float* rc = mtp_rc + size_t(pos)*RL;
            q.memcpy(kc, ckv_n, size_t(KVL)*4);
            q.memcpy(rc, kv + KVL, size_t(RL)*4);
            const int klm = int(C.key_len_mla);
            q.parallel_for(sycl::range<1>(size_t(NH)*RL), [=](sycl::id<1> i){
                const int h2 = int(i[0])/RL, d2 = int(i[0])%RL;
                qr[h2*RL + d2] = qb[h2*klm + NOPE + d2];
            });
            k_rope_norm(q, qr, NH, RL, RL, pos, C.rope_theta);
            k_rope_norm(q, rc, 1,  RL, RL, pos, C.rope_theta);
        }
        k_q6_gemv_multi_small(q, l.wk_b.dev, qb, C.key_len_mla, qc, KVL,
                              l.wk_b.K, KVL, l.wk_b.row_bytes, NH);
        k_mla_attend(q, qc, qr, mtp_kc, mtp_rc, oc, NH, pos + 1, KVL, RL, kq_scale, scratch);
        k_q6_gemv_multi_small(q, l.wv_b.dev, oc, KVL, ov, VHD,
                              l.wv_b.K, VHD, l.wv_b.row_bytes, NH);
        if (want_tp) {
            k_q6_gemv(q, l.wo0h.dev, ov, attn_o, l.wo0h.K, l.wo0h.N, l.wo0h.row_bytes);
            k_q6_gemv(q, l.wo0b.dev, ov + l.wo0h.K, wo_part, l.wo0b.K, l.wo0b.N, l.wo0b.row_bytes);
            k_add(q, attn_o, wo_part, H);
        } else {
            k_q6_gemv(q, l.wo.dev, ov, attn_o, l.wo.K, l.wo.N, l.wo.row_bytes);
        }
        k_add(q, mtp_cur, attn_o, H);
        k_add(q, mtp_cur, mtp_sa, H);            // residual
        q.memcpy(mtp_sa, mtp_cur, size_t(H)*4);  // ffn_inp

        // ---- MoE on the MTP block (single token, gpu0) ----
        k_rmsnorm(q, mtp_cur, l.ffn_norm, xb, H, C.rms_eps);
        k_q8_gemv(q, l.router.dev, xb, rlogits, l.router.K, l.router.N);
        q.memcpy(h_rlogits.data(), rlogits, size_t(NE)*4).wait();
        for (int e = 0; e < NE; ++e) {
            h_probs[e] = 1.f/(1.f + std::exp(-h_rlogits[e]));
            sel[e] = { h_probs[e] + l.h_bias[e], e };
        }
        std::partial_sort(sel.begin(), sel.begin()+TOPK, sel.end(),
                          [](auto&a, auto&b){ return a.first > b.first; });
        float wsum = 0.f, wts[64];
        for (int k = 0; k < TOPK; ++k) { wts[k] = h_probs[sel[k].second]; wsum += wts[k]; }
        for (int k = 0; k < TOPK; ++k) wts[k] = wts[k]/wsum*C.expert_weights_scale;

        EW& ew = EWs[0];
        Cache& c = CA[0];
        std::vector<sycl::event> dep;
        for (int k = 0; k < TOPK; ++k) {
            const int e = sel[k].second;
            const int64_t key = int64_t(NL)*NE + e;
            int sidx = c.lookup[size_t(key)];
            if (sidx < 0) {
                int victim = int(c.static_cap);
                for (int i = int(c.static_cap); i < int(c.slot.size()); ++i)
                    if (c.age[i] < c.age[victim]) victim = i;
                if (c.key[victim] >= 0) c.lookup[size_t(c.key[victim])] = -1;
                sidx = victim; c.key[sidx] = key; c.lookup[size_t(key)] = sidx;
                uint8_t* src = hot_base ? hot_ptr[size_t(key)] : nullptr;
                if (src) dep.push_back(c.qx.memcpy(c.slot[sidx], src, slot_bytes));
                else {
                    const size_t r = c.ring++ % size_t(STAGE_RING);
                    c.stage_ev[r].wait();
                    uint8_t* st = c.stage[r];
                    std::vector<PreadPool::Job> jb = {
                        {gfd, st,              l.off_gate + uint64_t(e)*gu_bytes, gu_bytes},
                        {gfd, st + gu_bytes,   l.off_up   + uint64_t(e)*gu_bytes, gu_bytes},
                        {gfd, st + 2*gu_bytes, l.off_down + uint64_t(e)*dn_bytes, dn_bytes}};
                    pool.run(jb);
                    c.stage_ev[r] = c.qx.memcpy(c.slot[sidx], st, slot_bytes);
                    dep.push_back(c.stage_ev[r]);
                }
            }
            if (c.age[sidx] != UINT64_MAX) c.age[sidx] = ++clk;
            ew.bases[k] = c.slot[sidx];
            ew.ub[k]    = c.slot[sidx] + gu_bytes;
            ew.db[k]    = c.slot[sidx] + 2*gu_bytes;
            ew.wts[k]   = wts[k];
        }
        k_iq2kt_gemv_rt<32,8>(q, ew.bases, xb, ew.gu,  H, EF, TOPK, dep);
        k_iq2kt_gemv_rt<32,8>(q, ew.ub,    xb, ew.out, H, EF, TOPK, {});
        k_silu_mul(q, ew.gu, ew.out, ew.gu, size_t(EF)*TOPK);
        k_iq2kt_moe_down(q, TT[0], ew.db, ew.gu, EF, ew.wts, ew.moe, EF, H, TOPK);
        k_q6_gemv(q, l.sh_gate.dev, xb, ffn_g, l.sh_gate.K, l.sh_gate.N, l.sh_gate.row_bytes);
        k_q6_gemv(q, l.sh_up.dev,   xb, ffn_u, l.sh_up.K,   l.sh_up.N,   l.sh_up.row_bytes);
        k_silu_mul(q, ffn_g, ffn_u, ffn_a, EF);
        if (want_shtp) {
            k_q6_gemv(q, l.shd0.dev,  ffn_a, attn_o, l.shd0.K, l.shd0.N, l.shd0.row_bytes);
            k_q6_gemv(q, l.shd0b.dev, ffn_a + l.shd0.K, wo_part, l.shd0b.K, l.shd0b.N, l.shd0b.row_bytes);
            k_add(q, attn_o, wo_part, H);
        } else {
            k_q6_gemv(q, l.sh_down.dev, ffn_a, attn_o, l.sh_down.K, l.sh_down.N, l.sh_down.row_bytes);
        }
        k_add(q, ew.moe, attn_o, H);
        k_add(q, mtp_cur, ew.moe, H);
        k_add(q, mtp_cur, mtp_sa, H);

        k_rmsnorm(q, mtp_cur, mtp_shnorm, xb, H, C.rms_eps);
        k_q6_gemv(q, lm_head.dev, xb, logits, lm_head.K, lm_head.N, lm_head.row_bytes);
        q.wait();
    };

    // ---- batched prefill: T tokens in one pass -----------------------------
    // Returns with `logits` holding the distribution for the LAST token, and the
    // KV cache filled for positions [0, T). Numerically the same graph as the
    // single-token path; only the batching differs.
    auto forward_chunk = [&](const std::vector<int>& toks, int pos0, bool all_logits = false) {
        const int T = int(toks.size());
        {   // embed all T rows on the host (one Q6_0 gather each)
            const auto* ti = g.find_tensor("token_embd.weight");
            const int64_t src_row = (int64_t(H)/32)*26;
            std::vector<float> hb(size_t(T)*H);
            for (int t = 0; t < T; ++t) {
                const uint8_t* rp = ti->data + int64_t(toks[t])*src_row;
                float* dst = hb.data() + size_t(t)*H;
                for (int b = 0; b < H/32; ++b) {
                    const uint8_t* blk = rp + b*26;
                    sycl::half dh; std::memcpy(&dh, blk, 2);
                    const float d = float(dh);
                    for (int j = 0; j < 16; ++j) {
                        const int l0 = blk[10+j] & 0x0F, l1 = blk[10+j] >> 4;
                        const int h0 = (blk[2 + (j%8)] >> (4*(j/8))) & 3;
                        const int h1 = (blk[2 + (j%8)] >> (4*(j/8)+2)) & 3;
                        dst[b*32 + j]      = d*float((l0 | (h0<<4)) - 32);
                        dst[b*32 + j + 16] = d*float((l1 | (h1<<4)) - 32);
                    }
                }
            }
            q.memcpy(Tx, hb.data(), size_t(T)*H*4).wait();
        }
        Clock::time_point pt = Clock::now();
        pphase(p_embed, pt);

        std::vector<float> hrl(size_t(T)*NE), hpb(NE);
        std::vector<std::vector<int32_t>> etok(NE);
        std::vector<std::vector<float>>   ewt(NE);
        std::vector<std::vector<int8_t>>  ekc(NE);   // canonical gather slot per (e,i)
        std::vector<int> eoff(NE, 0);

        for (int il = 0; il < NL; ++il) {
            Layer& l = L[il];
            k_rmsnorm_T(q, Tx, l.attn_norm, Txb, H, C.rms_eps, T);

            k_q6_gemm(q, l.wq_a.dev,  Txb, Tqa, l.wq_a.K,  l.wq_a.N,  l.wq_a.row_bytes, T);
            k_rmsnorm_T(q, Tqa, l.q_a_norm, Tqa, QL, C.rms_eps, T);
            k_q6_gemm(q, l.wkv_a.dev, Txb, Tkv, l.wkv_a.K, l.wkv_a.N, l.wkv_a.row_bytes, T);

            {   // split kv rows into the 512-d latent and the 64-d rope key
                const int kvl = KVL, rl = RL, kvr = KVL + RL;
                q.parallel_for(sycl::range<1>(size_t(T)*(KVL+RL)), [=](sycl::id<1> i){
                    const int t = int(i[0]) / kvr, j = int(i[0]) % kvr;
                    if (j < kvl) Tckv[int64_t(t)*kvl + j] = Tkv[int64_t(t)*kvr + j];
                    else         Tkr [int64_t(t)*rl + (j-kvl)] = Tkv[int64_t(t)*kvr + j];
                });
            }
            k_rmsnorm_T(q, Tckv, l.kv_a_norm, Tckv, KVL, C.rms_eps, T);

            k_q6_gemm(q, l.wq_b.dev, Tqa, Tqb, l.wq_b.K,
                      int64_t(NHT)*C.key_len_mla, l.wq_b.row_bytes, T);
            {   // gather q_rope out of the per-head [nope|rope] layout
                const int klm = int(C.key_len_mla), nope = NOPE, rl = RL, nh = NHT;
                q.parallel_for(sycl::range<1>(size_t(T)*NHT*RL), [=](sycl::id<1> i){
                    const int idx = int(i[0]);
                    const int t = idx / (nh*rl), r = idx % (nh*rl);
                    const int hh = r / rl, dd = r % rl;
                    Tqr[int64_t(t)*nh*rl + hh*rl + dd] =
                        Tqb[int64_t(t)*nh*klm + hh*klm + nope + dd];
                });
            }
            k_rope_norm_T(q, Tqr, NHT, RL, RL, pos0, C.rope_theta, T, int64_t(NHT)*RL);
            k_rope_norm_T(q, Tkr, 1,  RL, RL, pos0, C.rope_theta, T, RL);

            {   // append this chunk to the KV cache at [pos0, pos0+T)
                float* kc = kcache + size_t(il)*n_ctx*KVL + size_t(pos0)*KVL;
                float* rc = rcache + size_t(il)*n_ctx*RL  + size_t(pos0)*RL;
                q.memcpy(kc, Tckv, size_t(T)*KVL*4);
                q.memcpy(rc, Tkr,  size_t(T)*RL*4);
            }

            // ---- per-head work, split across both cards when TP is on -------
            // Everything from wq_b to wv_b is per-head, and the latent KV cache
            // both halves read is shared, so the only traffic a split costs is
            // q_lora + the KV slice out (21 MB) and gpu1's half of the attention
            // output back (67 MB). Without it, 13.7 s of a 28.5 s prefill runs on
            // one card while the other idles.
            const int64_t klmT = C.key_len_mla;
            if (want_tp) {
                sycl::queue& q1c = CA[1].qc;
                q.memcpy(Hqa.data(), Tqa,  size_t(T)*QL*4);
                q.memcpy(Hkv.data(), Tckv, size_t(T)*KVL*4);
                q.memcpy(Hkv.data() + size_t(T)*KVL, Tkr, size_t(T)*RL*4).wait();
                q1c.memcpy(T1qa,  Hqa.data(), size_t(T)*QL*4);
                q1c.memcpy(T1ckv, Hkv.data(), size_t(T)*KVL*4);
                q1c.memcpy(T1kr,  Hkv.data() + size_t(T)*KVL, size_t(T)*RL*4);
                q1c.memcpy(kcache1 + size_t(il)*n_ctx*KVL + size_t(pos0)*KVL,
                           T1ckv, size_t(T)*KVL*4);
                q1c.memcpy(rcache1 + size_t(il)*n_ctx*RL + size_t(pos0)*RL,
                           T1kr, size_t(T)*RL*4);
                k_q6_gemm(q1c, l.wq_b1.dev, T1qa, T1qb, l.wq_b1.K, l.wq_b1.N,
                          l.wq_b1.row_bytes, T);
                {   const int klm = int(klmT), nope = NOPE, rl = RL, nh = NHT;
                    q1c.parallel_for(sycl::range<1>(size_t(T)*NHT*RL), [=](sycl::id<1> i){
                        const int idx = int(i[0]);
                        const int t = idx / (nh*rl), r = idx % (nh*rl);
                        const int hh = r / rl, dd = r % rl;
                        T1qr[int64_t(t)*nh*rl + hh*rl + dd] =
                            T1qb[int64_t(t)*nh*klm + hh*klm + nope + dd];
                    });
                }
                k_rope_norm_T(q1c, T1qr, NHT, RL, RL, pos0, C.rope_theta, T, int64_t(NHT)*RL);
                k_q6_gemv_multi_small_T(q1c, l.wk_b1.dev, T1qb, klmT, int64_t(NHT)*klmT,
                                        T1qc, KVL, int64_t(NHT)*KVL,
                                        l.wk_b1.K, KVL, l.wk_b1.row_bytes, NHT, T);
                k_mla_flash_RG<8,128,4>(q1c, T1qc, T1qr, kcache1 + size_t(il)*n_ctx*KVL,
                                        rcache1 + size_t(il)*n_ctx*RL, T1oc, NHT, KVL, RL,
                                        kq_scale, T, pos0);
                k_q6_gemv_multi_small_T(q1c, l.wv_b1.dev, T1oc, KVL, int64_t(NHT)*KVL,
                                        T1ov, VHD, int64_t(NHT)*VHD,
                                        l.wv_b1.K, VHD, l.wv_b1.row_bytes, NHT, T);
            }
            k_q6_gemv_multi_small_T(q, l.wk_b.dev, Tqb, klmT,
                                    int64_t(NHT)*klmT, Tqc, KVL, int64_t(NHT)*KVL,
                                    l.wk_b.K, KVL, l.wk_b.row_bytes, NHT, T);

            if (g_gemm_v2 && (NHT % 8) == 0)
                k_mla_flash_RG<8,128,4>(q, Tqc, Tqr, kcache + size_t(il)*n_ctx*KVL,
                                        rcache + size_t(il)*n_ctx*RL, Toc, NHT, KVL, RL,
                                        kq_scale, T, pos0);
            else
                k_mla_flash_T(q, Tqc, Tqr, kcache + size_t(il)*n_ctx*KVL,
                              rcache + size_t(il)*n_ctx*RL, Toc, NHT, KVL, RL,
                              kq_scale, T, pos0);

            k_q6_gemv_multi_small_T(q, l.wv_b.dev, Toc, KVL, int64_t(NHT)*KVL,
                                    Tov, VHD, int64_t(NHT)*VHD,
                                    l.wv_b.K, VHD, l.wv_b.row_bytes, NHT, T);

            if (want_tp) {
                // Each card applies its own K-slice of wo to its own heads and
                // produces a PARTIAL output projection; summing the two partials
                // is exact, and it keeps wo's 31.4 TFLOP off a single card.
                k_q6_gemm(CA[1].qc, l.wo1h.dev, T1ov, T1att,
                          l.wo1h.K, l.wo1h.N, l.wo1h.row_bytes, T);
                k_q6_gemm(q, l.wo0h.dev, Tov, Tattn,
                          l.wo0h.K, l.wo0h.N, l.wo0h.row_bytes, T);
                CA[1].qc.memcpy(Hov.data(), T1att, size_t(T)*H*4).wait();
                q.memcpy(Tov1d, Hov.data(), size_t(T)*H*4);
                k_add(q, Tattn, Tov1d, int64_t(T)*H);
            } else {
                k_q6_gemm(q, l.wo.dev, Tov, Tattn, l.wo.K, l.wo.N, l.wo.row_bytes, T);
            }
            k_add(q, Tx, Tattn, int64_t(T)*H);

            pphase(p_attn, pt);
            k_rmsnorm_T(q, Tx, l.ffn_norm, Txb, H, C.rms_eps, T);
            if (C.is_dense_layer(uint32_t(il))) {
                k_q6_gemm(q, l.ffn_gate.dev, Txb, Tffg, l.ffn_gate.K, l.ffn_gate.N, l.ffn_gate.row_bytes, T);
                k_q6_gemm(q, l.ffn_up.dev,   Txb, Tffu, l.ffn_up.K,   l.ffn_up.N,   l.ffn_up.row_bytes, T);
                k_silu_mul_T(q, Tffg, Tffu, Tffa, int64_t(T)*C.ffn);
                k_q6_gemm(q, l.ffn_down.dev, Tffa, Tattn, l.ffn_down.K, l.ffn_down.N, l.ffn_down.row_bytes, T);
                k_add(q, Tx, Tattn, int64_t(T)*H);
                pphase(p_dense, pt);
                continue;
            }

            // ---- MoE, grouped by expert -------------------------------------
            if (T >= 64) k_q8_gemm_v3<64,64,32,4,4>(q, l.router.dev, Txb, Trl,
                                                    l.router.K, l.router.N, T);
            else         k_q8_gemm(q, l.router.dev, Txb, Trl, l.router.K, l.router.N, T);
            q.memcpy(Tbias, l.h_bias.data(), size_t(NE)*4);
            k_router_topk(q, Trl, Tbias, Ttopi, Ttopw, NE, TOPK,
                          C.expert_weights_scale, T);
            q.memcpy(Htopi.data(), Ttopi, size_t(T)*TOPK*4);
            // g_rp: the 50 MB xb D2H rides the router D2H's wait (in-order q),
            // and gpu1's H2D is issued immediately after — the separate
            // bounce-and-wait below disappears from the serial path.
            if (g_rp && NG > 1) q.memcpy(Tbounce.data(), Txb, size_t(T)*H*4);
            { auto epr = q.memcpy(Htopw.data(), Ttopw, size_t(T)*TOPK*4);
              if (g_probe) { const auto tpr = Clock::now(); spin_wait(epr); g_w_route += dsec(tpr); }
              else spin_wait(epr); }
            if (g_rp && NG > 1) CA[1].qc.memcpy(T1xb, Tbounce.data(), size_t(T)*H*4);

            for (int e = 0; e < NE; ++e) { etok[e].clear(); ewt[e].clear(); ekc[e].clear(); }
            for (int t = 0; t < T; ++t) {
                const int32_t* te = &Htopi[size_t(t)*TOPK];
                for (int k = 0; k < TOPK; ++k) {
                    const int e = te[k];
                    // Canonical gather slot: rank of e among this token's expert
                    // ids sorted ascending. That is exactly the order the old
                    // first-free fill produced under ascending-e processing, so
                    // the gather's float-summation order — and the output — stay
                    // byte-identical while the processing order becomes free.
                    int cs = 0;
                    for (int k2 = 0; k2 < TOPK; ++k2) cs += (te[k2] < e);
                    etok[e].push_back(t);
                    ewt [e].push_back(Htopw[size_t(t)*TOPK + k]);
                    ekc [e].push_back(int8_t(cs));
                }
            }

            const bool grouped = g_gemm_v2 && T >= (getenv("IE_GLM52_MOE_MIN_T")
                                  ? atoi(getenv("IE_GLM52_MOE_MIN_T")) : 2);
            k_zero(q, Tmoe, int64_t(T)*H);
            if (NG > 1) {
                if (!grouped) k_zero(CA[1].qc, T1moe, int64_t(T)*H);
                if (!g_rp) {
                    q.memcpy(Tbounce.data(), Txb, size_t(T)*H*4).wait();
                    CA[1].qc.memcpy(T1xb, Tbounce.data(), size_t(T)*H*4);
                }
            }
            // The shared expert depends only on the resident weights and Txb, not
            // on any routed expert, so running it HERE -- before the expert
            // fetches are waited on -- lets ~1.6 s of it hide behind the 3.0 s of
            // expert DMA that §16i showed does not overlap with anything else.
            // Its result sits in Tattn and is folded into Tmoe below, unchanged.
            if (want_tp && want_shtp) {
                const int64_t eh = EF/2;
                sycl::queue& q1c = CA[1].qc;
                k_q6_gemm(q1c, l.shg1.dev, T1xb, T1ffg, l.shg1.K, eh, l.shg1.row_bytes, T);
                k_q6_gemm(q1c, l.shu1.dev, T1xb, T1ffu, l.shu1.K, eh, l.shu1.row_bytes, T);
                k_silu_mul_T(q1c, T1ffg, T1ffu, T1ffg, int64_t(T)*eh);
                k_q6_gemm(q1c, l.shd1.dev, T1ffg, T1att, l.shd1.K, l.shd1.N, l.shd1.row_bytes, T);
                k_q6_gemm(q, l.sh_gate.dev, Txb, Tffg, l.sh_gate.K, eh, l.sh_gate.row_bytes, T);
                k_q6_gemm(q, l.sh_up.dev,   Txb, Tffu, l.sh_up.K,   eh, l.sh_up.row_bytes, T);
                k_silu_mul_T(q, Tffg, Tffu, Tffa, int64_t(T)*eh);
                k_q6_gemm(q, l.shd0.dev, Tffa, Tattn, l.shd0.K, l.shd0.N, l.shd0.row_bytes, T);
                CA[1].qc.memcpy(Hov.data(), T1att, size_t(T)*H*4).wait();
                q.memcpy(Tov1d, Hov.data(), size_t(T)*H*4);
                k_add(q, Tattn, Tov1d, int64_t(T)*H);
            } else {
            k_q6_gemm(q, l.sh_gate.dev, Txb, Tffg, l.sh_gate.K, l.sh_gate.N, l.sh_gate.row_bytes, T);
            k_q6_gemm(q, l.sh_up.dev,   Txb, Tffu, l.sh_up.K,   l.sh_up.N,   l.sh_up.row_bytes, T);
            k_silu_mul_T(q, Tffg, Tffu, Tffa, int64_t(T)*EF);
            k_q6_gemm(q, l.sh_down.dev, Tffa, Tattn, l.sh_down.K, l.sh_down.N, l.sh_down.row_bytes, T);
            }
            {   // pack every expert's token list, then one H2D per card
                int32_t* ht = (g_rp && (il & 1)) ? HalltB.data() : Hallt.data();
                float*   hw = (g_rp && (il & 1)) ? HallwB.data() : Hallw.data();
                if (g_rp) {
                    // this parity's buffer was last shipped in block il-2; its
                    // copies must land before the rewrite. Waiting on those two
                    // events replaces the full drain of both compute queues.
                    if (g_probe) { const auto tpr = Clock::now();
                                   hall_evq[il & 1].wait(); hall_ev1[il & 1].wait();
                                   g_w_top += dsec(tpr); }
                    else { hall_evq[il & 1].wait(); hall_ev1[il & 1].wait(); }
                }
                int off = 0;
                for (int e = 0; e < NE; ++e) {
                    eoff[e] = off;
                    for (size_t i = 0; i < etok[e].size(); ++i) {
                        ht[size_t(off)] = etok[e][i];
                        hw[size_t(off)] = ewt[e][i];
                        ++off;
                    }
                }
                q.memcpy(Tallt, ht, size_t(off)*4);
                hall_evq[il & 1] = q.memcpy(Tallw, hw, size_t(off)*4);
                if (NG > 1) {
                    CA[1].qc.memcpy(T1t, ht, size_t(off)*4);
                    hall_ev1[il & 1] = CA[1].qc.memcpy(T1w, hw, size_t(off)*4);
                }
                if (!g_rp) {
                    // Hallt/Hallw are rewritten by the next block and these copies
                    // are async, so they must land before the host moves on.
                    if (g_probe) { const auto tpr = Clock::now(); q.wait();
                                   if (NG > 1) CA[1].qc.wait(); g_w_top += dsec(tpr); }
                    else { q.wait(); if (NG > 1) CA[1].qc.wait(); }
                }
            }
            // Batched dispatch: residency is resolved for a GROUP of experts,
            // then the whole group runs in five launches instead of five per
            // expert. A group closes early only if a cache miss would have to
            // evict a slot that group's pending kernels are still going to read.
            // IE_GLM52_GEMM=0 keeps the original per-expert path for A/B.
            // Tile shape follows tokens-per-expert (about T/32): a 64-row tile
            // amortises the trellis decode 4x better, but at a 512-token chunk
            // an expert only has ~16 rows and 3/4 of it would be padding.
            pphase(p_route, pt);
            // The token tile must match the window. A 32-row tile against a
            // 2-token speculative verify wastes 32x of the FMA phase, which is
            // why the batched path lost on short inputs (1.70 s vs 0.78 s at
            // T=12) despite issuing 5 launches per block instead of 5 per expert.
            // An 8-row tile fixes the verify window without giving up the launch
            // saving that makes speculative decode viable at all.
            const int64_t tpe = int64_t(T)*TOPK/NE;        // tokens per expert
            const int MBM = (tpe >= 48) ? 64 : (T >= 32 ? 32 : 8);
            const int MOE_GROUP = g_moe_group;
            const int MOE_MIN_T = getenv("IE_GLM52_MOE_MIN_T")
                                  ? atoi(getenv("IE_GLM52_MOE_MIN_T")) : 2;
            if (g_gemm_v2 && T >= MOE_MIN_T) {
            struct Grp {
                std::vector<uint8_t*> base;
                std::vector<int32_t> eo, ec, go, tl, rtok;
                std::vector<float> rw;
                std::vector<sycl::event> dep;
                int rows = 0, r0 = 0;      // r0 = rows when this group opened
            };
            std::vector<Grp> GR{static_cast<size_t>(NG)};
            std::vector<uint64_t> gclk(size_t(NG), clk);
            std::vector<size_t> gidx(size_t(NG), 0);   // groups closed so far, per card
            for (int d = 0; d < NG; ++d) {
                GR[size_t(d)].rtok.assign(size_t(T)*TOPK, -1);
                GR[size_t(d)].rw.assign(size_t(T)*TOPK, 0.f);
            }
            auto launch = [&](int d) {
                Grp& G = GR[size_t(d)];
                if (G.base.empty()) return;
                Cache& c = CA[d];
                sycl::queue& qq = c.qc;
                MoEDesc& M = MD[size_t(d)];
                const int ne = int(G.base.size()), ntiles = int(G.tl.size()/2);
                const int sl = M.slot++ % MDSLOT;
                // The host runs ahead of the GPU, so a slot must not be rewritten
                // until its own copy has actually executed -- MDSLOT launches of
                // slack, not zero. Waiting on just this event keeps the pipeline
                // MDSLOT groups deep instead of draining the queue every group.
                if (g_probe) { const auto tpr = Clock::now(); M.ev[sl].wait();
                               g_w_mev += dsec(tpr); ++g_n_mev; }
                else M.ev[sl].wait();
                uint8_t* hs = M.hstage[sl];
                std::memcpy(hs + M.o_base, G.base.data(), size_t(ne)*sizeof(uint8_t*));
                std::memcpy(hs + M.o_eoff, G.eo.data(), size_t(ne)*4);
                std::memcpy(hs + M.o_ecnt, G.ec.data(), size_t(ne)*4);
                std::memcpy(hs + M.o_goff, G.go.data(), size_t(ne)*4);
                std::memcpy(hs + M.o_tile, G.tl.data(), size_t(ntiles)*2*4);
                // rtok/rw are read only by the per-block gather, which stages its
                // own copy below — staging them per GROUP was dead weight, and
                // DMAing all of M.bytes moved ~250 KB of stale tail per launch
                // (~96% of the descriptor at a 2-token verify window).
                uint8_t* ds = M.dstage[sl];
                M.ev[sl] = M.qd[0].memcpy(ds, hs, M.o_tile + size_t(ntiles)*2*4);
                G.dep.push_back(M.ev[sl]);
                if (probe_blk(il) && g_probe_chunks == 0)
                    g_probe_ev.push_back({d, il, 1, M.bytes, M.ev[sl]});
                uint8_t** dbase = reinterpret_cast<uint8_t**>(ds + M.o_base);
                const int32_t* deoff = reinterpret_cast<const int32_t*>(ds + M.o_eoff);
                const int32_t* decnt = reinterpret_cast<const int32_t*>(ds + M.o_ecnt);
                const int32_t* dgoff = reinterpret_cast<const int32_t*>(ds + M.o_goff);
                const int32_t* dtile = reinterpret_cast<const int32_t*>(ds + M.o_tile);
                const int32_t* drtok = reinterpret_cast<const int32_t*>(ds + M.o_rtok);
                const float*   drw   = reinterpret_cast<const float*>  (ds + M.o_rw);
                float* gu = (d==0 ? Tgub : T1gu);
                float* up = (d==0 ? Tupb : T1up);
                float* dn = (d==0 ? Tdnb : T1dn);
                const float* xsrc = (d==0 ? Txb : T1xb);
                const int32_t* tka = (d==0 ? Tallt : T1t);
                float* mout = (d==0 ? Tmoe : T1moe);
                auto three = [&](auto tag) {
                    constexpr int BM = decltype(tag)::bm, TM = decltype(tag)::tm;
                    // A fused gate+up kernel (k_iq2kt_moe_gemm2) is 1.19x on the
                    // pair in isolation and a NET LOSS in situ: MoE -0.57 s but
                    // attention +4.66 s, pp 91.88 -> 77.31. Its 512-lane
                    // work-groups appear to interfere with the cross-card
                    // attention wait. Kept in the file, unused.
                    auto pe1 = k_iq2kt_moe_gemm<BM,256,32,TM,8,true>(qq, TT[size_t(d)], dbase, deoff, decnt, dgoff,
                            dtile, ntiles, 0, tka, xsrc, H, gu, H, EF, G.dep);
                    auto pe2 = k_iq2kt_moe_gemm<BM,256,32,TM,8,true>(qq, TT[size_t(d)], dbase, deoff, decnt, dgoff,
                            dtile, ntiles, gu_bytes, tka, xsrc, H, up, H, EF, {});
                    k_silu_mul_T(qq, gu + int64_t(G.r0)*EF, up + int64_t(G.r0)*EF,
                                 gu + int64_t(G.r0)*EF, int64_t(G.rows - G.r0)*EF);
                    auto pe3 = k_iq2kt_moe_gemm<BM,256,32,TM,8,true>(qq, TT[size_t(d)], dbase, deoff, decnt, dgoff,
                            dtile, ntiles, 2*gu_bytes, nullptr, gu, EF, dn, EF, H, {});
                    c.last_moe = pe3;
                    if (probe_blk(il) && g_probe_chunks == 0) {
                        g_probe_ev.push_back({d, il, 2, 0, pe1});
                        g_probe_ev.push_back({d, il, 2, 0, pe2});
                        g_probe_ev.push_back({d, il, 2, 0, pe3});
                    }
                };
                if (MBM == 64)      three(MoeTile<64,8>{});
                else if (MBM == 32) three(MoeTile<32,4>{});
                else                three(MoeTile<8,1>{});
                // The gather-add is NOT done here. Running it per group made it
                // run once per group over the whole T x H grid -- eight groups
                // cost ~1 s per chunk and cancelled exactly the overlap the
                // groups were there to buy, which is why small groups measured
                // worse. Rows now accumulate across the block and one gather
                // runs after the last group.
                (void)drtok; (void)drw; (void)mout;
                G.base.clear(); G.eo.clear(); G.ec.clear(); G.go.clear();
                G.tl.clear(); G.dep.clear(); G.r0 = G.rows;
                // gclk is NOT advanced here: these kernels are still in flight
                // (there is no longer a drain), so every slot this block has
                // touched must stay un-evictable until something waits.
            };
            // With the top-of-block drain gone (g_rp), nothing guarantees the
            // PREVIOUS block's GEMMs have finished reading the slots this
            // block's misses may evict. Order the eviction copies after them on
            // the transfer queue — free on gpu0 (the in-order main queue already
            // sequenced them behind this block's attention), and exactly the
            // required ordering on gpu1.
            if (g_rp)
                for (int d2 = 0; d2 < NG; ++d2)
                    CA[d2].qx.ext_oneapi_submit_barrier(
                        std::vector<sycl::event>{CA[d2].last_moe});
            // Hits-first: every resident expert is grouped and launched BEFORE
            // any miss is resolved, so the hit GEMMs — which have no copy
            // dependency — execute UNDER the miss copy stream instead of after
            // it. Side benefit: a miss can no longer evict a slot a later hit in
            // the same block was about to use.
            int eorder[256]; int ne_act = 0;
            if (g_hits_first) {
                for (int pass = 0; pass < 2; ++pass)
                    for (int e = 0; e < NE; ++e) {
                        if (etok[e].empty()) continue;
                        const int64_t key = int64_t(il)*NE + e;
                        const int od = (NG > 1) ? int(owner[size_t(key)]) : 0;
                        if ((CA[od].lookup[size_t(key)] >= 0) == (pass == 0))
                            eorder[ne_act++] = e;
                    }
            } else {
                for (int e = 0; e < NE; ++e)
                    if (!etok[e].empty()) eorder[ne_act++] = e;
            }
            bool miss_seen[2] = {false, false};
            const int MISS_GROUP = getenv("IE_GLM52_MISS_GROUP")
                                   ? atoi(getenv("IE_GLM52_MISS_GROUP")) : 64;
            for (int ei = 0; ei < ne_act; ++ei) {
                const int e = eorder[ei];
                const int nt = int(etok[e].size());
                const int64_t key = int64_t(il)*NE + e;
                const int od = (NG > 1) ? int(owner[size_t(key)]) : 0;
                Cache& c = CA[od];
                Grp& G = GR[size_t(od)];
                int sidx = c.lookup[size_t(key)];
                if (sidx < 0 && !miss_seen[size_t(od)]) {
                    // hit->miss boundary for this card: flush the pure-hit group
                    // so its GEMMs start with no copy dependency at all.
                    miss_seen[size_t(od)] = true;
                    if (g_hits_first && !G.base.empty()) { launch(od); ++gidx[size_t(od)]; }
                }
                if (sidx < 0) {
                    int victim = -1;
                    for (int i = int(c.static_cap); i < int(c.slot.size()); ++i)
                        if (c.age[i] <= gclk[size_t(od)] &&
                            (victim < 0 || c.age[i] < c.age[victim])) victim = i;
                    if (victim < 0) {           // reservoir exhausted by this block
                        launch(od);
                        if (g_probe) { const auto tpr = Clock::now(); c.qc.wait();
                                       g_w_drain += dsec(tpr); ++g_n_drain; }
                        else c.qc.wait();       // slots must be free before reuse
                        gclk[size_t(od)] = clk; // ...and only now are they
                        for (int i = int(c.static_cap); i < int(c.slot.size()); ++i)
                            if (victim < 0 || c.age[i] < c.age[victim]) victim = i;
                    }
                    if (c.key[victim] >= 0) c.lookup[size_t(c.key[victim])] = -1;
                    sidx = victim; c.key[sidx] = key; c.lookup[size_t(key)] = sidx;
                    ++c.misses;
                    uint8_t* src = hot_base ? hot_ptr[size_t(key)] : nullptr;
                    if (src) { ++c.pin_hits;
                               G.dep.push_back(c.qx.memcpy(c.slot[sidx], src, slot_bytes));
                               if (probe_blk(il) && g_probe_chunks == 0)
                                   g_probe_ev.push_back({od, il, 0, slot_bytes, G.dep.back()}); }
                    else {
                        const size_t r = c.ring++ % size_t(STAGE_RING);
                        c.stage_ev[r].wait();
                        uint8_t* stg = c.stage[r];
                        std::vector<PreadPool::Job> jb = {
                            {gfd, stg,              l.off_gate + uint64_t(e)*gu_bytes, gu_bytes},
                            {gfd, stg + gu_bytes,   l.off_up   + uint64_t(e)*gu_bytes, gu_bytes},
                            {gfd, stg + 2*gu_bytes, l.off_down + uint64_t(e)*dn_bytes, dn_bytes}};
                        pool.run(jb);
                        c.stage_ev[r] = c.qx.memcpy(c.slot[sidx], stg, slot_bytes);
                        G.dep.push_back(c.stage_ev[r]);
                    }
                    c.h2d += slot_bytes;
                } else ++c.hits;
                if (c.age[sidx] != UINT64_MAX) c.age[sidx] = ++clk;
                const int ke = int(G.base.size());
                G.base.push_back(c.slot[sidx]);
                G.eo.push_back(int32_t(eoff[e]));
                G.ec.push_back(int32_t(nt));
                G.go.push_back(int32_t(G.rows));
                for (int m0 = 0; m0 < nt; m0 += MBM) { G.tl.push_back(ke); G.tl.push_back(m0); }
                for (int i = 0; i < nt; ++i) {
                    const int t  = etok[e][size_t(i)];
                    const int cs = ekc[e][size_t(i)];
                    G.rtok[size_t(t)*TOPK + cs] = int32_t(G.rows + i);
                    G.rw  [size_t(t)*TOPK + cs] = ewt[e][size_t(i)];
                }
                G.rows += nt;
                c.nk_total += nt; ++c.nk_blocks;
                // Hit experts accumulate into big groups (MOE_GROUP): they have
                // no copy dependency, so the first launch happens at the
                // hit->miss boundary above and runs under the whole miss copy
                // stream. Miss groups stay small (MISS_GROUP) so each one's GEMM
                // starts as soon as ITS copies land instead of the whole block's.
                const size_t gi = gidx[size_t(od)];
                const int cap = g_hits_first
                                ? (miss_seen[size_t(od)] ? MISS_GROUP : MOE_GROUP)
                                : ((gi == 0) ? 8 : (gi == 1) ? 24 : MOE_GROUP);
                if (int(G.base.size()) >= cap) { launch(od); ++gidx[size_t(od)]; }
            }
            pphase(p_fetch, pt);        // residency resolution + DMA issue
            for (int d = 0; d < NG; ++d) launch(d);
            for (int d = 0; d < NG; ++d) {          // one gather per card per block
                Grp& G = GR[size_t(d)];
                if (G.rows == 0) continue;
                Cache& c = CA[d];
                MoEDesc& M = MD[size_t(d)];
                sycl::queue& qq = c.qc;
                const int sl = M.slot++ % MDSLOT;
                M.ev[sl].wait();
                uint8_t* hs = M.hstage[sl];
                std::memcpy(hs + M.o_rtok, G.rtok.data(), size_t(T)*TOPK*4);
                std::memcpy(hs + M.o_rw,   G.rw.data(),   size_t(T)*TOPK*4);
                uint8_t* ds = M.dstage[sl];
                M.qd[0].memcpy(ds + M.o_rtok, hs + M.o_rtok, size_t(T)*TOPK*4);
                M.ev[sl] = M.qd[0].memcpy(ds + M.o_rw, hs + M.o_rw, size_t(T)*TOPK*4);
                float* dn = (d==0 ? Tdnb : T1dn);
                float* mout = (d==0 ? Tmoe : T1moe);
                const int32_t* rt = reinterpret_cast<const int32_t*>(ds + M.o_rtok);
                const float*  rwp = reinterpret_cast<const float*>  (ds + M.o_rw);
                const int hh = H, tk2 = TOPK;
                // gpu0 accumulates into its zeroed device buffer (+= over 0 ==
                // plain store, byte-identical); gpu1 STORES into host memory —
                // '=' avoids the PCIe read a '+=' would need, and every (t,j)
                // is written exactly once by this single gather.
                float* mo1 = (d == 1 && T1moeH) ? T1moeH : mout;
                const bool hostout = (d == 1 && T1moeH);
                auto gev = qq.submit([&](sycl::handler& h2){
                    h2.depends_on(M.ev[sl]);
                    h2.parallel_for(sycl::range<1>(size_t(T)*H), [=](sycl::id<1> i){
                        const int idx = int(i[0]);
                        const int t = idx / hh, j = idx % hh;
                        float sa = 0.f;
                        for (int k2 = 0; k2 < tk2; ++k2) {
                            const int r = rt[t*tk2 + k2];
                            if (r >= 0) sa += rwp[t*tk2 + k2]*dn[int64_t(r)*hh + j];
                        }
                        if (hostout) mo1[int64_t(t)*hh + j] = sa;
                        else         mo1[int64_t(t)*hh + j] += sa;
                    });
                });
                if (d == 1) CA[1].last_moe = gev;
            }
            } else {
            int gu_off = 0, gu_off1 = 0;
            for (int e = 0; e < NE; ++e) {
                const int nt = int(etok[e].size());
                if (!nt) continue;
                const int64_t key = int64_t(il)*NE + e;
                const int od = (NG > 1) ? int(owner[size_t(key)]) : 0;
                Cache& c = CA[od];
                int sidx = c.lookup[size_t(key)];
                std::vector<sycl::event> dep;
                if (sidx < 0) {
                    int victim = int(c.static_cap);
                    for (int i = int(c.static_cap); i < int(c.slot.size()); ++i)
                        if (c.age[i] < c.age[victim]) victim = i;
                    if (c.key[victim] >= 0) c.lookup[size_t(c.key[victim])] = -1;
                    sidx = victim; c.key[sidx] = key; c.lookup[size_t(key)] = sidx;
                    uint8_t* src = hot_base ? hot_ptr[size_t(key)] : nullptr;
                    if (src) dep.push_back(c.qx.memcpy(c.slot[sidx], src, slot_bytes));
                    else {
                        const size_t r = c.ring++ % size_t(STAGE_RING);
                        c.stage_ev[r].wait();
                        uint8_t* stg = c.stage[r];
                        std::vector<PreadPool::Job> jb = {
                            {gfd, stg,              l.off_gate + uint64_t(e)*gu_bytes, gu_bytes},
                            {gfd, stg + gu_bytes,   l.off_up   + uint64_t(e)*gu_bytes, gu_bytes},
                            {gfd, stg + 2*gu_bytes, l.off_down + uint64_t(e)*dn_bytes, dn_bytes}};
                        pool.run(jb);
                        c.stage_ev[r] = c.qx.memcpy(c.slot[sidx], stg, slot_bytes);
                        dep.push_back(c.stage_ev[r]);
                    }
                }
                if (c.age[sidx] != UINT64_MAX) c.age[sidx] = ++clk;
                uint8_t* base = c.slot[sidx];
                sycl::queue& qq = c.qc;
                int32_t* tk  = (od==0 ? Tallt : T1t) + eoff[e];
                float*   wt  = (od==0 ? Tallw : T1w) + eoff[e];
                int&     goff = (od==0 ? gu_off : gu_off1);
                float* xsrc = (od==0 ? Txb  : T1xb);
                float* mout = (od==0 ? Tmoe : T1moe);
                int32_t* sq = (od==0 ? Tseq : T1seq);
                float* gu = (od==0 ? Tgub : T1gu) + int64_t(goff)*EF;
                float* up = (od==0 ? Tupb : T1up) + int64_t(goff)*EF;
                float* dn = (od==0 ? Tdnb : T1dn) + int64_t(goff)*H;
                if (use_xmx) {
                    k_iq2kt_gemm_xmx(qq, base,            xsrc, H, tk, nt, gu, H,  EF, dep);
                    k_iq2kt_gemm_xmx(qq, base + gu_bytes, xsrc, H, tk, nt, up, H,  EF, {});
                    k_silu_mul_T(qq, gu, up, gu, int64_t(nt)*EF);
                    k_iq2kt_gemm_xmx(qq, base + 2*gu_bytes, gu, EF, sq, nt, dn, EF, H, {});
                } else {
                    k_iq2kt_gemm(qq, base,              xsrc, H, tk, nt, gu, H,  EF, dep);
                    k_iq2kt_gemm(qq, base + gu_bytes,   xsrc, H, tk, nt, up, H,  EF, {});
                    k_silu_mul_T(qq, gu, up, gu, int64_t(nt)*EF);
                    k_iq2kt_gemm(qq, base + 2*gu_bytes, gu,  EF, sq, nt, dn, EF, H, {});
                }
                {   // scatter-add w[i] * dn[i] into that card's MoE accumulator
                    const int hh = H;
                    qq.parallel_for(sycl::range<1>(size_t(nt)*H), [=](sycl::id<1> i){
                        const int idx = int(i[0]);
                        const int r = idx / hh, j = idx % hh;
                        mout[int64_t(tk[r])*hh + j] += wt[r]*dn[int64_t(r)*hh + j];
                    });
                }
                goff += nt;
            }
            }
            // Start gpu1's MoE partial moving NOW but fold it in AFTER the shared
            // expert: the shared expert does not depend on it, so ~100 MB of
            // round-trip per block (0.6 s per chunk) hides behind ~1.4 s of work
            // instead of blocking it.
            sycl::event fold_ev;
            const bool zfold = (NG > 1) && grouped && T1moeH;
            if (NG > 1) fold_ev = zfold ? CA[1].last_moe
                                        : CA[1].qc.memcpy(Tbounce.data(), T1moe, size_t(T)*H*4);
            pphase(p_moe, pt);
            k_add(q, Tmoe, Tattn, int64_t(T)*H);
            if (NG > 1) {   // now fold gpu1's partial in
                if (g_probe) { const auto tpr = Clock::now(); spin_wait(fold_ev); g_w_fold += dsec(tpr); }
                else spin_wait(fold_ev);
                q.memcpy(Tfold, zfold ? T1moeH : Tbounce.data(), size_t(T)*H*4);
                k_add(q, Tmoe, Tfold, int64_t(T)*H);
            }
            k_add(q, Tx, Tmoe, int64_t(T)*H);
            pphase(p_dense, pt);
        }
        if (all_logits) {
            // Verification needs the distribution after EVERY drafted token, not
            // just the last one. Tlog is sized for the verify window only.
            if (T > TLOGR) { std::fprintf(stderr, "verify window %d > Tlog rows %d\n", T, TLOGR); std::exit(1); }
            k_rmsnorm_T(q, Tx, out_norm, Txb, H, C.rms_eps, T);
            k_q6_gemm(q, lm_head.dev, Txb, Tlog, lm_head.K, lm_head.N, lm_head.row_bytes, T);
        } else {
            k_rmsnorm(q, Tx + int64_t(T-1)*H, out_norm, xb, H, C.rms_eps);
            k_q6_gemv(q, lm_head.dev, xb, logits, lm_head.K, lm_head.N, lm_head.row_bytes);
        }
        q.wait();
        pphase(p_head, pt);
        if (g_probe && g_probe_chunks++ == 0) probe_dump("prefill");
    };

    auto forward_prefill = [&](const std::vector<int>& toks) {
        for (size_t off = 0; off < toks.size(); off += size_t(TB)) {
            const size_t n = std::min<size_t>(size_t(TB), toks.size() - off);
            forward_chunk({toks.begin() + off, toks.begin() + off + n}, int(off));
        }
    };

    // ---- forward for one token at position `pos` ---------------------------
    auto forward = [&](int token, int pos) {
        // embedding: dequantise one Q6_0 row of token_embd on the host
        {
            const auto* ti = g.find_tensor("token_embd.weight");
            const int64_t src_row = (int64_t(H)/32)*26;
            const uint8_t* rp = ti->data + int64_t(token)*src_row;
            for (int b = 0; b < H/32; ++b) {
                const uint8_t* blk = rp + b*26;
                sycl::half dh; std::memcpy(&dh, blk, 2);
                const float d = float(dh);
                for (int j = 0; j < 16; ++j) {
                    const int l0 = blk[10+j] & 0x0F, l1 = blk[10+j] >> 4;
                    const int h0 = (blk[2 + (j%8)] >> (4*(j/8))) & 3;
                    const int h1 = (blk[2 + (j%8)] >> (4*(j/8)+2)) & 3;
                    h_embed[b*32 + j]      = d*float((l0 | (h0<<4)) - 32);
                    h_embed[b*32 + j + 16] = d*float((l1 | (h1<<4)) - 32);
                }
            }
            q.memcpy(x, h_embed.data(), size_t(H)*4).wait();
        }

        const int n_kv = pos + 1;
        Clock::time_point tp = Clock::now();
        Clock::time_point hp = Clock::now();
        auto seg = [&](double& a){ if (g_hostprof) { a += dsec(hp); hp = Clock::now(); } };
        for (int il = 0; il < NL; ++il) {
            Layer& l = L[il];
            if (g_hostprof) hp = Clock::now();
            k_rmsnorm(q, x, l.attn_norm, xb, H, C.rms_eps);

            // ---- MLA ----
            k_q6_gemv(q, l.wq_a.dev, xb, qa, l.wq_a.K, l.wq_a.N, l.wq_a.row_bytes);
            k_rmsnorm(q, qa, l.q_a_norm, qa, QL, C.rms_eps);
            k_q6_gemv(q, l.wq_b.dev, qa, qb, l.wq_b.K, l.wq_b.N, l.wq_b.row_bytes);
            k_q6_gemv(q, l.wkv_a.dev, xb, kv, l.wkv_a.K, l.wkv_a.N, l.wkv_a.row_bytes);
            k_rmsnorm(q, kv, l.kv_a_norm, ckv_n, KVL, C.rms_eps);

            // split q into per-head nope/rope; rope both halves
            float* kc = kcache + (size_t(il)*n_ctx + pos)*KVL;
            float* rc = rcache + (size_t(il)*n_ctx + pos)*RL;
            q.memcpy(kc, ckv_n, size_t(KVL)*4);
            q.memcpy(rc, kv + KVL, size_t(RL)*4);
            {   // gather q_rope out of the interleaved [nope|rope] per-head layout
                const int klm = int(C.key_len_mla);
                q.parallel_for(sycl::range<1>(size_t(NH)*RL), [=](sycl::id<1> i){
                    const int h = int(i[0])/RL, d = int(i[0])%RL;
                    qr[h*RL + d] = qb[h*klm + NOPE + d];
                });
            }
            k_rope_norm(q, qr, NH, RL, RL, pos, C.rope_theta);
            k_rope_norm(q, rc, 1, RL, RL, pos, C.rope_theta);

            // qc[h] = wk_b[h]^T . q_nope[h].
            // wk_b is [192, 512, 64]: 64 slices of 512 rows x 192 cols. q_nope[h]
            // is qb[h*key_len_mla .. +192], already contiguous because
            // key_len_mla = nope(192) + rope(64) and nope comes first — so the
            // batched GEMV can read it in place with stride key_len_mla.
            k_q6_gemv_multi_small(q, l.wk_b.dev, qb, C.key_len_mla, qc, KVL,
                                  l.wk_b.K, KVL, l.wk_b.row_bytes, NH);

            k_mla_attend(q, qc, qr, kcache + size_t(il)*n_ctx*KVL,
                         rcache + size_t(il)*n_ctx*RL, oc, NH, n_kv, KVL, RL,
                         kq_scale, scratch);

            // ov[h] = wv_b[h]^T . oc[h].  wv_b is [512, 256, 64].
            k_q6_gemv_multi_small(q, l.wv_b.dev, oc, KVL, ov, VHD,
                                  l.wv_b.K, VHD, l.wv_b.row_bytes, NH);

            if (want_tp) {
                k_q6_gemv(q, l.wo0h.dev, ov, attn_o, l.wo0h.K, l.wo0h.N, l.wo0h.row_bytes);
                k_q6_gemv(q, l.wo0b.dev, ov + l.wo0h.K, wo_part, l.wo0b.K, l.wo0b.N, l.wo0b.row_bytes);
                k_add(q, attn_o, wo_part, H);
            } else {
                k_q6_gemv(q, l.wo.dev, ov, attn_o, l.wo.K, l.wo.N, l.wo.row_bytes);
            }
            k_add(q, x, attn_o, H);
            phase(t_attn, tp);

            // ---- FFN ----
            k_rmsnorm(q, x, l.ffn_norm, xb, H, C.rms_eps);
            if (C.is_dense_layer(uint32_t(il))) {
                k_q6_gemv(q, l.ffn_gate.dev, xb, ffn_g, l.ffn_gate.K, l.ffn_gate.N, l.ffn_gate.row_bytes);
                k_q6_gemv(q, l.ffn_up.dev,   xb, ffn_u, l.ffn_up.K,   l.ffn_up.N,   l.ffn_up.row_bytes);
                k_silu_mul(q, ffn_g, ffn_u, ffn_a, C.ffn);
                k_q6_gemv(q, l.ffn_down.dev, ffn_a, attn_o, l.ffn_down.K, l.ffn_down.N, l.ffn_down.row_bytes);
                k_add(q, x, attn_o, H);
                continue;
            }

            seg(hp_attn);
            // router
            k_q8_gemv(q, l.router.dev, xb, rlogits, l.router.K, l.router.N);
            if (g_dec2 && ng_use > 1) {
                // Ship xb to gpu1 under the SAME host sync as the router logits:
                // one D2H wait per block instead of two, and gpu1's H2D rides
                // under the host-side top-k + residency work below.
                q.memcpy(h_rlogits.data(), rlogits, size_t(NE)*4);
                spin_wait(q.memcpy(bounce.data(), xb, size_t(H)*4));
                CA[1].qc.memcpy(EWs[1].xb, bounce.data(), size_t(H)*4);
            } else
            spin_wait(q.memcpy(h_rlogits.data(), rlogits, size_t(NE)*4));
            if (g_dec2) {
                // Shared expert as early as possible: it depends only on xb and
                // resident weights, so it computes while the host does top-k +
                // residency and while the miss copies fly.
                k_q6_gemv(q, l.sh_gate.dev, xb, ffn_g, l.sh_gate.K, l.sh_gate.N, l.sh_gate.row_bytes);
                k_q6_gemv(q, l.sh_up.dev,   xb, ffn_u, l.sh_up.K,   l.sh_up.N,   l.sh_up.row_bytes);
                k_silu_mul(q, ffn_g, ffn_u, ffn_a, EF);
                k_q6_gemv(q, l.sh_down.dev, ffn_a, attn_o, l.sh_down.K, l.sh_down.N, l.sh_down.row_bytes);
            }
            phase(t_router, tp);
            // sigmoid -> add exp_probs_b for SELECTION ONLY -> top-k -> gather the
            // UNBIASED probs as weights -> normalise -> scale. Exactly
            // llm_build_moe_ffn (ik src/llama-build-context.cpp:1488-1560).
            for (int e = 0; e < NE; ++e) {
                h_probs[e] = 1.f/(1.f + std::exp(-h_rlogits[e]));
                sel[e] = { h_probs[e] + l.h_bias[e], e };
            }
            std::partial_sort(sel.begin(), sel.begin()+TOPK, sel.end(),
                              [](auto&a, auto&b){ return a.first > b.first; });
            // Phase 9 router trace: one line per (token, block) with the chosen
            // expert ids. This is the ground truth the cache-policy and prefetch
            // work needs — everything measured so far assumed a synthetic zipf.
            if (trace) {
                std::fprintf(trace, "%d %d", pos, il);
                for (int k = 0; k < TOPK; ++k) std::fprintf(trace, " %d", sel[k].second);
                std::fputc('\n', trace);
            }
            float wsum = 0.f; float wts[64];
            for (int k = 0; k < TOPK; ++k) { wts[k] = h_probs[sel[k].second]; wsum += wts[k]; }
            for (int k = 0; k < TOPK; ++k) wts[k] = wts[k]/wsum*C.expert_weights_scale;
            seg(hp_route);

            // Each GPU fetches and computes its own slice of the top-k. Both
            // slices are issued before either is waited on, so the two PCIe links
            // and the two compute engines run concurrently.
            // Ownership decides WHERE an expert is cached, but a block can hand
            // one card 8 of the 8 and the other none. Mean was 4.04/3.96, yet the
            // per-block variance puts one card on the critical path while the
            // other idles — worth ~7 ms/token. So honour ownership only up to an
            // even split, then spill the excess to the other card (which fetches
            // it from the host tier instead of its own VRAM).
            // MEASURED NEGATIVE: perfect 4.00/4.00 compute balance cost more in
            // fetch than it saved in compute (misses 13,716 -> 17,350, tier
            // coverage 88.6%/81.9% -> 72.5%/68.6%, decode 5.11 -> 3.32 tok/s).
            // Respecting ownership — even with a lumpy 0..8 split — wins, because
            // an expert on its owner's card is a VRAM hit and anywhere else is a
            // PCIe transfer. Kept behind a flag rather than deleted.
            int assign[64];
            const bool balance = getenv("IE_GLM52_BALANCE") != nullptr;
            if (ng_use > 1 && !balance) {
                for (int k = 0; k < TOPK; ++k)
                    assign[k] = owner[size_t(int64_t(il)*NE + sel[k].second)];
            } else if (ng_use > 1) {
                const int cap = (TOPK + ng_use - 1) / ng_use;
                int cnt[2] = {0, 0};
                for (int k = 0; k < TOPK; ++k) assign[k] = -1;
                for (int k = 0; k < TOPK; ++k) {          // first pass: owner wins
                    const int o = owner[size_t(int64_t(il)*NE + sel[k].second)];
                    if (cnt[o] < cap) { assign[k] = o; ++cnt[o]; }
                }
                for (int k = 0; k < TOPK; ++k)             // spill the rest
                    if (assign[k] < 0) {
                        const int o = cnt[0] <= cnt[1] ? 0 : 1;
                        assign[k] = o; ++cnt[o];
                    }
            }
            // An expert slot is [gate | up | down], but `down` is not needed until
            // after the gate/up GEMV. Splitting the fetch in two lets the down
            // third (3.4 of 10.1 MB) transfer WHILE gate/up compute, which is the
            // only part of decode's 81 ms/token fetch that anything can hide.
            static std::vector<std::vector<sycl::event>> deps, deps_dn;
            deps.assign(size_t(ng_use), {}); deps_dn.assign(size_t(ng_use), {});
            static std::vector<PreadPool::Job> pj;
            for (int d = 0; d < ng_use; ++d) {
                EW& ew = EWs[d];
                Cache& c = CA[d];
                const int ns = int(c.slot.size());
                // Route by OWNERSHIP, not by top-k position. Each (block,expert)
                // belongs to exactly one card, so the two VRAM caches partition
                // the expert space instead of both caching the same hot experts.
                // Counts vary per block (0..8, ~4 on average); ew.nk records it.
                ew.nk = 0;
                if (g_descstage) {
                    // claim this block's ring slot; its previous copy is long done
                    ew.dslot = (ew.dslot + 1) % 8;
                    uint8_t* hb = ew.hblob[ew.dslot];
                    ew.hb_bases = reinterpret_cast<uint8_t**>(hb);
                    ew.hb_ub    = ew.hb_bases + TOPK;
                    ew.hb_db    = ew.hb_ub + TOPK;
                    ew.hb_wts   = reinterpret_cast<float*>(ew.hb_db + TOPK);
                }
                for (int k = 0; k < TOPK; ++k) {
                    const int e = sel[k].second;
                    if (ng_use > 1 && assign[k] != d) continue;
                    const int64_t key = int64_t(il)*NE + e;
                    // Level 5: one in-order queue orders copy-before-kernel by
                    // construction, so no event is retained. (Prefetch, which is
                    // the only other producer of an in-flight fill, is measured
                    // negative and off.)
                    const bool noev = g_dec_lvl >= 5;
                    bool kmiss = false; sycl::event kev, kev_dn;
                    int s = c.lookup[size_t(key)];
                    if (g_stream && s < 0) {
                        // MISS, STREAMED: hand the kernel the host pointer. No
                        // slot is claimed, nothing is copied, nothing is waited
                        // on — the GEMV reads the weights over PCIe as it runs.
                        uint8_t* hp = hot_ptr[size_t(key)];
                        if (hp) {
                            ++c.misses; ++c.pin_hits;
                            c.h2d += slot_bytes;
                            const int j2 = ew.nk++;
                            if (g_descstage) {
                                ew.hb_bases[j2] = hp; ew.hb_ub[j2] = hp + gu_bytes;
                                ew.hb_db[j2] = hp + 2*gu_bytes; ew.hb_wts[j2] = wts[k];
                            } else {
                            ew.bases[j2] = hp;
                            ew.ub[j2]    = hp + gu_bytes;
                            ew.db[j2]    = hp + 2*gu_bytes;
                            ew.wts[j2]   = wts[k];
                            }
                            ew.mflag[size_t(j2)] = 1;
                            continue;
                        }
                    }
                    if (s >= 0) {
                        ++c.hits;
                        // May have been filled by a prefetch that is still in
                        // flight; the kernel must wait for it.
                        if (!noev && !g_leanev) { deps[d].push_back(c.fill[s]);
                                                  kev = c.fill[s]; kev_dn = c.fill[s]; }
                    } else {
                        ++c.misses; kmiss = true;
                        int victim = -1;
                        if (g_clock && ns > int(c.static_cap)) {
                            // Pure FIFO rotation — genuinely O(1). The first
                            // attempt at this used the monotonic age counter as
                            // CLOCK's reference bit, but ages are essentially
                            // never 0, so the "second chance" pass zeroed the
                            // whole span and it stayed O(n): resolve 34-62
                            // ms/token instead of ~1 (run24). With 27.9%
                            // token-to-token reuse, FIFO and LRU are close in
                            // quality anyway.
                            const int lo = int(c.static_cap), span = ns - lo;
                            if (int(c.hand) < lo || int(c.hand) >= ns) c.hand = size_t(lo);
                            for (int tries = 0; tries < span; ++tries) {
                                const int i = int(c.hand);
                                c.hand = size_t(lo + (i - lo + 1) % span);
                                if (c.age[i] == UINT64_MAX) continue;      // pinned
                                victim = i; break;
                            }
                            if (victim < 0) victim = lo;
                        } else {
                        for (int i = int(c.static_cap); i < ns; ++i)
                            if (victim < 0 || c.age[i] < c.age[victim]) victim = i;
                        }
                        if (victim < 0) victim = int(c.static_cap);   // paranoia
                        if (c.key[victim] >= 0) c.lookup[size_t(c.key[victim])] = -1;
                        s = victim; c.key[s] = key; c.lookup[size_t(key)] = s;
                        // Reserve a staging buffer; the pread jobs for this and
                        // every other miss in this block are issued together
                        // below so the pool stays saturated.
                        uint8_t* src = hot_base ? hot_ptr[size_t(key)] : nullptr;
                        if (src) {
                            // Already in the pinned hot tier: DMA straight from it.
                            // Split shape for BOTH decode variants: the gate/up
                            // GEMV is gated by 2/3 of the bytes, and the down
                            // third transfers underneath it. A single full-slot
                            // copy put the whole 10 MB on the GEMV's critical
                            // path and lost 15-23% end to end (run2/run3).
                            ++c.pin_hits;
                            sycl::queue& qx2 = (g_dec_lvl == 4) ? c.qc
                                               : c.qxs[c.qxi++ % g_qx];
                            const auto tcp = Clock::now();
                            if (noev) {
                                // one transfer, no event retained: the split
                                // existed only to gate two kernels separately.
                                qx2.memcpy(c.slot[s], src, slot_bytes);
                                if (g_hostprof) hp_r_cpy += dsec(tcp);
                            } else if (g_onecopy) {
                                c.fill[s] = qx2.memcpy(c.slot[s], src, slot_bytes);
                                if (g_hostprof) hp_r_cpy += dsec(tcp);
                                deps[d].push_back(c.fill[s]);
                                kev = c.fill[s]; kev_dn = c.fill[s];
                            } else {
                            deps[d].push_back(qx2.memcpy(c.slot[s], src, 2*gu_bytes));
                            c.fill[s] = qx2.memcpy(c.slot[s] + 2*gu_bytes,
                                                   src + 2*gu_bytes, dn_bytes);
                            if (g_hostprof) hp_r_cpy += dsec(tcp);
                            deps_dn[d].push_back(c.fill[s]);
                            kev = deps[d].back(); kev_dn = c.fill[s];
                            if (probe_blk(il) && g_probe_calls >= 3 && g_probe_calls < 6) {
                                g_probe_ev.push_back({d, il, 0, 2*gu_bytes, deps[d].back()});
                                g_probe_ev.push_back({d, il, 0, dn_bytes, c.fill[s]});
                            }
                            }
                        } else {
                            // Stage through the pinned ring, and promote this
                            // expert into the hot tier while budget remains, so
                            // the NEXT time it is wanted the copy is skipped.
                            const auto tdk = Clock::now();
                            ++c.disk_reads;
                            const size_t r = c.ring++ % size_t(STAGE_RING);
                            c.stage_ev[r].wait();      // its previous DMA must be done
                            uint8_t* st = c.stage[r];
                            pj.clear();
                            pj.push_back({gfd, st,              l.off_gate + uint64_t(e)*gu_bytes, gu_bytes});
                            pj.push_back({gfd, st + gu_bytes,   l.off_up   + uint64_t(e)*gu_bytes, gu_bytes});
                            pj.push_back({gfd, st + 2*gu_bytes, l.off_down + uint64_t(e)*dn_bytes, dn_bytes});
                            pool.run(pj);
                            // stage_ev is still retained: the staging ring waits
                            // on it before reusing the buffer.
                            c.stage_ev[r] = ((g_dec_lvl == 4) ? c.qc : c.qx)
                                                .memcpy(c.slot[s], st, slot_bytes);
                            c.disk_secs += dsec(tdk);
                            if (!noev) {
                                c.fill[s] = c.stage_ev[r];
                                deps[d].push_back(c.stage_ev[r]);
                                kev = c.stage_ev[r]; kev_dn = c.stage_ev[r];
                            }
                            uint8_t* hot = (hot_base && hot_used + slot_bytes <= hot_cap)
                                           ? hot_alloc(slot_bytes) : nullptr;
                            if (hot) {
                                std::memcpy(hot, st, slot_bytes);
                                hot_ptr[size_t(key)] = hot;
                                // The tier now holds this expert, so the file
                                // pages behind it are pure duplication. Dropping
                                // them is what keeps tier + page cache inside RAM;
                                // skipping this is exactly how the 182 GiB pin,
                                // and later a 64 GiB one, pushed the box into
                                // thrashing.
                                posix_fadvise(gfd, off_t(l.off_gate + uint64_t(e)*gu_bytes),
                                              off_t(gu_bytes), POSIX_FADV_DONTNEED);
                                posix_fadvise(gfd, off_t(l.off_up + uint64_t(e)*gu_bytes),
                                              off_t(gu_bytes), POSIX_FADV_DONTNEED);
                                posix_fadvise(gfd, off_t(l.off_down + uint64_t(e)*dn_bytes),
                                              off_t(dn_bytes), POSIX_FADV_DONTNEED);
                            }
                        }
                        c.h2d += slot_bytes;
                    }
                    if (c.age[s] != UINT64_MAX) c.age[s] = ++clk;   // keep pinned rows pinned
                    const int j = ew.nk++;
                    if (g_descstage) {
                        ew.hb_bases[j] = c.slot[s]; ew.hb_ub[j] = c.slot[s] + gu_bytes;
                        ew.hb_db[j] = c.slot[s] + 2*gu_bytes; ew.hb_wts[j] = wts[k];
                    } else {
                    ew.bases[j] = c.slot[s];
                    ew.ub[j]    = c.slot[s] + gu_bytes;
                    ew.db[j]    = c.slot[s] + 2*gu_bytes;
                    ew.wts[j]   = wts[k];
                    }
                    ew.mflag[size_t(j)] = kmiss ? 1 : 0;
                    if (!noev && !g_leanev) { ew.fev[size_t(j)]    = kev;
                                              ew.fev_dn[size_t(j)] = kev_dn; }
                }
            }
            if (g_dec_lvl >= 5) {
                // The dependency the kernels no longer carry, paid here instead:
                // one host wait per card per block, on work that was never
                // overlapped anyway.
                for (int d = 0; d < ng_use; ++d) {
                    CA[d].qx.wait();
                    for (auto& qq2 : CA[d].qxs) qq2.wait();
                }
            }
            seg(hp_resolve);
            for (int d = 0; d < ng_use; ++d) { CA[d].nk_total += uint64_t(EWs[d].nk); ++CA[d].nk_blocks; }
            for (int k = 0; k < TOPK; ++k) prev_sel[size_t(il)][size_t(k)] = sel[k].second;
            if (PROF) { for (int d=0; d<ng_use; ++d) { CA[d].qx.wait();
                            for (auto& qq2 : CA[d].qxs) qq2.wait(); }
                        t_fetch += dsec(tp); tp = Clock::now(); }

            // GPU1 needs the post-ffn_norm activation; 24 KiB per MoE block.
            // (v2 shipped it under the router sync above.)
            if (!g_dec2 && ng_use > 1) {
                spin_wait(q.memcpy(bounce.data(), xb, size_t(H)*4));
                CA[1].qc.memcpy(EWs[1].xb, bounce.data(), size_t(H)*4);
            }
            seg(hp_bounce);

            // Shared expert BEFORE the routed ones. It needs only resident
            // weights and xb, so putting it here lets it execute while the
            // expert DMA issued above is still in flight -- the same
            // transformation that was worth 1.43 tok/s in prefill (§21a). The
            // adds below are unchanged, so the arithmetic is identical.
            // (v2 submits it even earlier, under the router sync.)
            if (!g_dec2) {
            auto se1 = k_q6_gemv(q, l.sh_gate.dev, xb, ffn_g, l.sh_gate.K, l.sh_gate.N, l.sh_gate.row_bytes);
            k_q6_gemv(q, l.sh_up.dev,   xb, ffn_u, l.sh_up.K,   l.sh_up.N,   l.sh_up.row_bytes);
            k_silu_mul(q, ffn_g, ffn_u, ffn_a, EF);
            auto se2 = k_q6_gemv(q, l.sh_down.dev, ffn_a, attn_o, l.sh_down.K, l.sh_down.N, l.sh_down.row_bytes);
            if (probe_blk(il) && g_probe_calls >= 3 && g_probe_calls < 6) {
                g_probe_ev.push_back({0, il, 3, 0, se1});
                g_probe_ev.push_back({0, il, 3, 0, se2});
            }
            }
            seg(hp_shared);
            for (int d = 0; d < ng_use; ++d) {
                EW& ew = EWs[d];
                const int nk = ew.nk;
                if (nk == 0) { k_zero(CA[d].qc, ew.moe, H); continue; }
                sycl::queue& qq = CA[d].qc;
                if (g_dec2 && g_dsplit) {
                    // Hit/miss split: hit GEMVs launch gated only by their
                    // (usually complete) fill events, so they compute while the
                    // miss copies are still in flight; miss GEMVs follow, gated
                    // by their own copies. The per-expert down + ascending-slot
                    // gather keeps the summation order identical to the fused
                    // kernel, so the output is byte-exact.
                    int nh = 0, nm = 0;
                    for (int k = 0; k < nk; ++k) if (!ew.mflag[size_t(k)]) {
                        ew.b2[nh] = ew.bases[k]; ew.u2[nh] = ew.ub[k]; ew.d2[nh] = ew.db[k];
                        ew.w2[nh] = ew.wts[k];   ew.smap[nh] = k; ++nh; }
                    for (int k = 0; k < nk; ++k) if (ew.mflag[size_t(k)]) {
                        ew.b2[nh+nm] = ew.bases[k]; ew.u2[nh+nm] = ew.ub[k]; ew.d2[nh+nm] = ew.db[k];
                        ew.w2[nh+nm] = ew.wts[k];   ew.smap[nh+nm] = k; ++nm; }
                    // NOTE: an event-status pre-filter (get_info per event) was
                    // tried here and cost ~100+ us PER CALL in-engine — hostprof
                    // measured the launch section at 160 ms/token with it.
                    static std::vector<sycl::event> dh, dm, dmdn;
                    dh.clear(); dm.clear(); dmdn.clear();
                    for (int k = 0; k < nk; ++k) {
                        (ew.mflag[size_t(k)] ? dm : dh).push_back(ew.fev[size_t(k)]);
                        if (ew.mflag[size_t(k)]) dmdn.push_back(ew.fev_dn[size_t(k)]);
                    }
                    sycl::event ge1, ge2;
                    if (nh) {
                        ge1 = k_iq2kt_gemv_rt<32,8>(qq, ew.b2, ew.xb, ew.gu,  H, EF, nh, dh);
                        k_iq2kt_gemv_rt<32,8>(qq, ew.u2, ew.xb, ew.out, H, EF, nh, {});
                        k_silu_mul(qq, ew.gu, ew.out, ew.gu, size_t(EF)*nh);
                        k_iq2kt_moe_down_pe(qq, TT[size_t(d)], ew.d2, ew.gu, EF,
                                            ew.w2, ew.smap, ew.dnrows, EF, H, nh);
                    }
                    if (nm) {
                        k_iq2kt_gemv_rt<32,8>(qq, ew.b2+nh, ew.xb, ew.gu+size_t(nh)*EF, H, EF, nm, dm);
                        k_iq2kt_gemv_rt<32,8>(qq, ew.u2+nh, ew.xb, ew.out+size_t(nh)*EF, H, EF, nm, {});
                        k_silu_mul(qq, ew.gu+size_t(nh)*EF, ew.out+size_t(nh)*EF,
                                   ew.gu+size_t(nh)*EF, size_t(EF)*nm);
                        if (!dmdn.empty()) qq.ext_oneapi_submit_barrier(dmdn);
                        k_iq2kt_moe_down_pe(qq, TT[size_t(d)], ew.d2+nh, ew.gu+size_t(nh)*EF, EF,
                                            ew.w2+nh, ew.smap+nh, ew.dnrows, EF, H, nm);
                    }
                    ge2 = k_moe_gather_rows(qq, ew.dnrows, ew.moe, H, nk);
                    if (probe_blk(il) && g_probe_calls >= 3 && g_probe_calls < 6) {
                        if (nh) g_probe_ev.push_back({d, il, 2, 0, ge1});
                        g_probe_ev.push_back({d, il, 2, 0, ge2});
                    }
                    continue;
                }
                const auto tl0 = Clock::now();
                uint8_t* const* kb = ew.bases; uint8_t* const* ku = ew.ub;
                uint8_t* const* kd = ew.db;     const float* kw = ew.wts;
                if (g_descstage) {
                    // The host writes TOPK-strided slots, so copy the whole blob
                    // regardless of nk; strides must match what the kernel reads.
                    const size_t nb2 = size_t(TOPK)*sizeof(uint8_t*);
                    uint8_t* db2 = ew.dblob[ew.dslot];
                    ew.dev_ev[ew.dslot] =
                        qq.memcpy(db2, ew.hblob[ew.dslot], ew.blob_bytes);
                    kb = reinterpret_cast<uint8_t* const*>(db2);
                    ku = reinterpret_cast<uint8_t* const*>(db2 + nb2);
                    kd = reinterpret_cast<uint8_t* const*>(db2 + 2*nb2);
                    kw = reinterpret_cast<const float*>(db2 + 3*nb2);
                }
                auto ge1 = k_iq2kt_gemv_rt<32,8>(qq, kb, ew.xb, ew.gu,  H, EF, nk,
                                                 g_dec_lvl >= 4 ? std::vector<sycl::event>{} : deps[d]);
                if (g_hostprof) hp_l_dep1 += dsec(tl0);
                const auto tl1 = Clock::now();
                k_iq2kt_gemv_rt<32,8>(qq, ku,    ew.xb, ew.out, H, EF, nk, {});
                k_silu_mul(qq, ew.gu, ew.out, ew.gu, size_t(EF)*nk);
                if (g_dec_lvl < 4 && !deps_dn[d].empty()) qq.ext_oneapi_submit_barrier(deps_dn[d]);
                auto ge2 = k_iq2kt_moe_down(qq, TT[size_t(d)], kd, ew.gu, EF, kw, ew.moe, EF, H, nk);
                if (g_hostprof) hp_l_rest += dsec(tl1);
                if (probe_blk(il) && g_probe_calls >= 3 && g_probe_calls < 6) {
                    g_probe_ev.push_back({d, il, 2, 0, ge1});
                    g_probe_ev.push_back({d, il, 2, 0, ge2});
                }
            }
            seg(hp_launch);
            // ---- predictive prefetch (Phase 11) -----------------------------
            // Decode's expert fetch is LATENCY bound, not bandwidth bound: the
            // same host tier sustains 22.8 GB/s per card under prefill's deep
            // queue and 13.8 under decode's, where ~1.5 transfers per block are
            // issued and immediately waited on. The routing of block il+1 is not
            // known yet, but the PREVIOUS token's routing for that block predicts
            // it well -- that correlation is exactly what gives the cache its 68%
            // hit rate. Issuing those fetches now puts them under this block's
            // GEMVs instead of in front of the next block's.
            if (do_prefetch && il + 1 < NL) {
                const size_t nx = size_t(il + 1);
                for (int k = 0; k < TOPK; ++k) {
                    const int e = prev_sel[nx][size_t(k)];
                    if (e < 0) continue;
                    const int64_t key = int64_t(nx)*NE + e;
                    const int d2 = (ng_use > 1) ? int(owner[size_t(key)]) : 0;
                    Cache& c2 = CA[d2];
                    if (c2.lookup[size_t(key)] >= 0) continue;      // already resident
                    uint8_t* src2 = hot_base ? hot_ptr[size_t(key)] : nullptr;
                    if (!src2) continue;                            // never prefetch off disk
                    const int ns2 = int(c2.slot.size());
                    int v2 = -1;
                    for (int i = int(c2.static_cap); i < ns2; ++i)
                        if (v2 < 0 || c2.age[i] < c2.age[v2]) v2 = i;
                    if (v2 < 0) continue;
                    if (c2.key[v2] >= 0) c2.lookup[size_t(c2.key[v2])] = -1;
                    c2.key[v2] = key; c2.lookup[size_t(key)] = v2;
                    c2.age[v2] = ++clk;
                    c2.fill[v2] = c2.qx.memcpy(c2.slot[v2], src2, slot_bytes);
                    c2.h2d += slot_bytes; ++c2.pf_issued;
                }
            }
            seg(hp_pf);
            if (ng_use > 1) {
                spin_wait(CA[1].qc.memcpy(bounce.data(), EWs[1].moe, size_t(H)*4));
                q.memcpy(moe_partial, bounce.data(), size_t(H)*4);
                k_add(q, EWs[0].moe, moe_partial, H);
            }
            seg(hp_fold);
            phase(t_exp, tp);
            k_add(q, EWs[0].moe, attn_o, H);      // shared expert, computed above
            k_add(q, x, EWs[0].moe, H);
            phase(t_shexp, tp);
        }
        k_rmsnorm(q, x, out_norm, xb, H, C.rms_eps);
        auto e_head = k_q6_gemv(q, lm_head.dev, xb, logits, lm_head.K, lm_head.N, lm_head.row_bytes);
        // Drain EVERY queue once per token. Their events are otherwise never
        // synchronized, and the UR adapter's pools grow until its append-paths
        // spin (hostprof: resolve 54->91 ms/token over 300 tokens, stacks in
        // clock_gettime/sched_yield under appendUSMMemcpy). Cleanup runs inside
        // real queue waits — event-status polling (spin_wait) bypasses it. All
        // work is complete by end of token, so these waits are ~free.
        for (int d2 = 0; d2 < ng_use; ++d2) {
            CA[d2].qc.wait();
            CA[d2].qx.wait();
            for (auto& qq2 : CA[d2].qxs) qq2.wait();
        }
        if (PROF) q.wait(); else spin_wait(e_head);
        if (g_hostprof) {
            uint64_t tot = 0;
            for (int d2 = 0; d2 < ng_use; ++d2) tot += CA[d2].h2d;
            hp_bytes = tot;
        }
        if (g_hostprof && ++hp_tokens % 50 == 0)
            std::printf("[hostprof %llu tok] ms/tok: attn-submit %.2f route %.2f resolve %.2f "
                        "(cpy %.2f) bounce %.2f shared %.2f launch %.2f (dep1 %.2f rest %.2f) "
                        "prefetch %.2f fold %.2f\n",
                        (unsigned long long)hp_tokens,
                        hp_attn/hp_tokens*1e3, hp_route/hp_tokens*1e3, hp_resolve/hp_tokens*1e3,
                        hp_r_cpy/hp_tokens*1e3,
                        hp_bounce/hp_tokens*1e3, hp_shared/hp_tokens*1e3, hp_launch/hp_tokens*1e3,
                        hp_l_dep1/hp_tokens*1e3, hp_l_rest/hp_tokens*1e3,
                        hp_pf/hp_tokens*1e3, hp_fold/hp_tokens*1e3),
            std::printf("            fetched %.1f MB/tok | implied over (resolve+launch) %.1f GB/s"
                        " of 53.1 peak\n",
                        double(hp_bytes)/hp_tokens/1e6,
                        double(hp_bytes)/1e9 / ((hp_resolve + hp_launch) > 0
                                                ? (hp_resolve + hp_launch) : 1));
        phase(t_head, tp);
        if (g_probe && ++g_probe_calls == 6) probe_dump("decode");
    };

    // ---- token strings for detokenisation ----------------------------------
    const auto* tk = g.find_kv("tokenizer.ggml.tokens");
    auto toks = tk->as_string_array();

    // ---- run ---------------------------------------------------------------
    std::vector<float> hl(VOC);
    std::vector<int> out_ids;
    std::printf("prompt: %zu tokens\n", prompt.size());

    // Measure every GPU configuration from this one load. The 27-minute cost of
    // populating the pinned expert bank dominates everything else, so paying it
    // once and sweeping is the only sane way to A/B the dual-GPU split.
    const int cfg_lo = getenv("IE_GLM52_ONLY_LAST") ? NG : 1;
    // (n_gpus, gemm variant). IE_GLM52_GEMM_SWEEP measures the SLM-blocked v2
    // GEMMs against the row-per-work-group v1 pair from the same load, because
    // the load is what costs half an hour, not the run.
    const int ndraft_env = getenv("IE_GLM52_NDRAFT") ? atoi(getenv("IE_GLM52_NDRAFT")) : 3;
    if (const char* e = getenv("IE_GLM52_MOE_GROUP")) g_moe_group = atoi(e);
    if (const char* e = getenv("IE_GLM52_HITS_FIRST")) g_hits_first = atoi(e);
    if (const char* e = getenv("IE_GLM52_DEC")) g_dec_lvl = atoi(e);
    g_dec2 = g_dec_lvl >= 2; g_dsplit = g_dec_lvl == 2;
    // The hit/miss split path gates miss GEMVs on per-slot fill events; a
    // streamed miss has no fill event because it has no copy. They are mutually
    // exclusive by construction.
    if (g_stream) g_dsplit = 0;
    if (const char* e = getenv("IE_GLM52_RP"))  g_rp = atoi(e);
    if (const char* e = getenv("IE_GLM52_STATIC_PCT")) g_static_pct = atoi(e);
    if (const char* e = getenv("IE_GLM52_STATIC_DEC")) g_static_pct_dec = atoi(e);
    if (const char* e = getenv("IE_GLM52_SPIN")) g_spin = atoi(e);
    if (const char* e = getenv("IE_GLM52_STREAM")) g_stream = atoi(e);
    g_hostprof = getenv("IE_GLM52_HOSTPROF") != nullptr;
    if (const char* e = getenv("IE_GLM52_STREAM")) g_stream = atoi(e);
    if (const char* e = getenv("IE_GLM52_ONECOPY")) g_onecopy = atoi(e);
    if (const char* e = getenv("IE_GLM52_LEANEV")) g_leanev = atoi(e);
    if (const char* e = getenv("IE_GLM52_DESCSTAGE")) g_descstage = atoi(e);
    if (const char* e = getenv("IE_GLM52_CLOCK")) g_clock = atoi(e);
    // [15] = prompt tokens for this config (0 = whole prompt). pp has a maximum
    // in T: the expert-bank sweep is one fixed cost per chunk while attention is
    // O(T^2), so throughput peaks partway (§21g put it near T=1888 for the old
    // implementation; hits-first shrank the constant term, so re-locate it).
    std::vector<std::array<int,17>> runs;   // +[16] = shared-expert TP split   // +[13] = staged descriptors  // +[10] = decode static%, -1 = unchanged
    for (int cfg = cfg_lo; cfg <= NG; ++cfg) {
        std::vector<int> gv;
        if (getenv("IE_GLM52_GEMM_SWEEP")) { gv.push_back(1); gv.push_back(0); }
        else gv.push_back(g_gemm_v2);
        for (int v : gv) {
            // IE_GLM52_MTP_SWEEP measures plain decode against 1- and 2-token
            // speculative windows from the SAME load: the verify cost that made
            // MTP a net loss was 4 launches per expert, and it is now 5 per block.
            if (getenv("IE_GLM52_PPOPT_SWEEP")) {
                // prefill only, one knob at a time; [16] = shared-expert TP
                runs.push_back({cfg, v, ndraft_env, 128, 1, 1, 3, g_rp, 87, 0, -1, 0, 0, 1, 0, 0, 0});
                runs.push_back({cfg, v, ndraft_env, 128, 1, 1, 3, g_rp, 87, 0, -1, 0, 0, 1, 0, 0, 1});
                runs.push_back({cfg, v, ndraft_env, 256, 1, 1, 3, g_rp, 87, 0, -1, 0, 0, 1, 0, 0, 0});
                runs.push_back({cfg, v, ndraft_env,  64, 1, 1, 3, g_rp, 87, 0, -1, 0, 0, 1, 0, 0, 0});
            } else if (getenv("IE_GLM52_PP_SWEEP")) {
                for (int T : {2000, 1888, 1750, 1500})
                    runs.push_back({cfg, v, ndraft_env, g_moe_group, 1, 1, 3, g_rp, 87, 0, -1,
                                    0, 0, 1, 0, T, g_shtp});
            } else if (getenv("IE_GLM52_CLOCK_SWEEP")) {
                // best known config, then trade static residency for hit rate
                // now that eviction is O(1)
                runs.push_back({cfg, v, ndraft_env, g_moe_group, 1, 1, 3, g_rp, 87, 0, -1, 0, 0, 1, 0, 0, g_shtp});
                runs.push_back({cfg, v, ndraft_env, g_moe_group, 1, 1, 3, g_rp, 87, 0, 20, 0, 0, 1, 1, 0, g_shtp});
                runs.push_back({cfg, v, ndraft_env, g_moe_group, 1, 1, 3, g_rp, 87, 0,  5, 0, 0, 1, 1, 0, g_shtp});
                runs.push_back({cfg, v, ndraft_env, g_moe_group, 1, 1, 3, g_rp, 87, 0, -1, 0, 0, 1, 0, 0, g_shtp});
            } else if (getenv("IE_GLM52_DESC_SWEEP")) {
                runs.push_back({cfg, v, ndraft_env, g_moe_group, 1, 1, 3, g_rp, 87, 0, -1, 0, 0, 0, g_clock, 0, g_shtp});
                runs.push_back({cfg, v, ndraft_env, g_moe_group, 1, 1, 3, g_rp, 87, 0, -1, 0, 0, 1, g_clock, 0, g_shtp});
                runs.push_back({cfg, v, ndraft_env, g_moe_group, 1, 1, 3, g_rp, 87, 0, -1, 0, 1, 1, g_clock, 0, g_shtp});
                runs.push_back({cfg, v, ndraft_env, g_moe_group, 1, 1, 3, g_rp, 87, 0, -1, 0, 0, 0, g_clock, 0, g_shtp});
            } else if (getenv("IE_GLM52_LEANEV_SWEEP")) {
                runs.push_back({cfg, v, ndraft_env, g_moe_group, 1, 1, 3, g_rp, 87, 0, -1, 0, 0, g_descstage, g_clock, 0, g_shtp});
                runs.push_back({cfg, v, ndraft_env, g_moe_group, 1, 1, 3, g_rp, 87, 0, -1, 0, 1, g_descstage, g_clock, 0, g_shtp});
                runs.push_back({cfg, v, ndraft_env, g_moe_group, 1, 1, 3, g_rp, 87, 0, -1, 1, 1, g_descstage, g_clock, 0, g_shtp});
                runs.push_back({cfg, v, ndraft_env, g_moe_group, 1, 1, 3, g_rp, 87, 0, -1, 0, 0, g_descstage, g_clock, 0, g_shtp});
            } else if (getenv("IE_GLM52_XING_SWEEP")) {
                // boundary-crossings per block, one variable at a time
                runs.push_back({cfg, v, ndraft_env, g_moe_group, 1, 1, 3, g_rp, 87, 0, -1, 0, g_leanev, g_descstage, g_clock, 0, g_shtp});
                runs.push_back({cfg, v, ndraft_env, g_moe_group, 1, 1, 3, g_rp, 87, 0, -1, 1, g_leanev, g_descstage, g_clock, 0, g_shtp});
                runs.push_back({cfg, v, ndraft_env, g_moe_group, 4, 1, 3, g_rp, 87, 0, -1, 1, g_leanev, g_descstage, g_clock, 0, g_shtp});
                runs.push_back({cfg, v, ndraft_env, g_moe_group, 1, 1, 5, g_rp, 87, 0, -1, 0, g_leanev, g_descstage, g_clock, 0, g_shtp});
            } else if (getenv("IE_GLM52_SDEC_SWEEP")) {
                // Prefill keeps its 87% static residency in every config; only
                // what decode gets handed afterwards varies. -1 = today's
                // behaviour (decode inherits prefill's split).
                runs.push_back({cfg, v, ndraft_env, g_moe_group, int(g_qx), 1, 3, g_rp, 87, 0, -1, g_onecopy, g_leanev, g_descstage, g_clock, 0, g_shtp});
                runs.push_back({cfg, v, ndraft_env, g_moe_group, int(g_qx), 1, 3, g_rp, 87, 0, 50, g_onecopy, g_leanev, g_descstage, g_clock, 0, g_shtp});
                runs.push_back({cfg, v, ndraft_env, g_moe_group, int(g_qx), 1, 3, g_rp, 87, 0, 20, g_onecopy, g_leanev, g_descstage, g_clock, 0, g_shtp});
                runs.push_back({cfg, v, ndraft_env, g_moe_group, int(g_qx), 1, 3, g_rp, 87, 0,  5, g_onecopy, g_leanev, g_descstage, g_clock, 0, g_shtp});
            } else if (getenv("IE_GLM52_STREAM_SWEEP")) {
                // one load, one variable at a time: copy-vs-stream at the same
                // static split, then stream with more static residency.
                runs.push_back({cfg, v, ndraft_env, g_moe_group, int(g_qx), 1, 3, g_rp, 87, 0, g_static_pct_dec, g_onecopy, g_leanev, g_descstage, g_clock, 0, g_shtp});
                runs.push_back({cfg, v, ndraft_env, g_moe_group, int(g_qx), 1, 3, g_rp, 87, 1, g_static_pct_dec, g_onecopy, g_leanev, g_descstage, g_clock, 0, g_shtp});
                runs.push_back({cfg, v, ndraft_env, g_moe_group, int(g_qx), 1, 3, g_rp, 95, 1, g_static_pct_dec, g_onecopy, g_leanev, g_descstage, g_clock, 0, g_shtp});
                runs.push_back({cfg, v, ndraft_env, g_moe_group, int(g_qx), 1, 3, g_rp, 60, 0, g_static_pct_dec, g_onecopy, g_leanev, g_descstage, g_clock, 0, g_shtp});
            } else if (getenv("IE_GLM52_MG_SWEEP")) {
                // miss-group size sweep: hits-first made fine-grained miss
                // groups unnecessary for overlap, and small groups cost GEMM
                // efficiency (run4: kernel busy 105 -> 116 ms/block at MG=32).
                for (int mg : {32, 64, 96, 128})
                    runs.push_back({cfg, v, ndraft_env, mg + (1<<16), int(g_qx), 1, g_dec_lvl, g_rp, g_static_pct, g_stream, g_static_pct_dec, g_onecopy, g_leanev, g_descstage, g_clock, 0, g_shtp});
            } else if (getenv("IE_GLM52_R3_SWEEP")) {
                // R3a prefill lift + decode v1 vs v2, same load, all hits-first
                runs.push_back({cfg, v, ndraft_env, g_moe_group, int(g_qx), 1, 0, 0, g_static_pct, g_stream, g_static_pct_dec, g_onecopy, g_leanev, g_descstage, g_clock, 0, g_shtp});
                runs.push_back({cfg, v, ndraft_env, g_moe_group, int(g_qx), 1, 0, 1, g_static_pct, g_stream, g_static_pct_dec, g_onecopy, g_leanev, g_descstage, g_clock, 0, g_shtp});
                runs.push_back({cfg, v, ndraft_env, g_moe_group, int(g_qx), 1, 1, 1, g_static_pct, g_stream, g_static_pct_dec, g_onecopy, g_leanev, g_descstage, g_clock, 0, g_shtp});
            } else if (getenv("IE_GLM52_P1_SWEEP")) {
                // old scheduling vs hits-first (and old vs v2 decode), same load
                runs.push_back({cfg, v, ndraft_env, g_moe_group, int(g_qx), 0, 0, g_rp, g_static_pct, g_stream, g_static_pct_dec, g_onecopy, g_leanev, g_descstage, g_clock, 0, g_shtp});
                runs.push_back({cfg, v, ndraft_env, g_moe_group, int(g_qx), 1, g_dec_lvl, g_rp, g_static_pct, g_stream, g_static_pct_dec, g_onecopy, g_leanev, g_descstage, g_clock, 0, g_shtp});
            } else if (getenv("IE_GLM52_DECSTATIC_SWEEP")) {
                // decode structure first (static fixed), then static split with
                // the structure fixed — each config differs in ONE variable.
                runs.push_back({cfg, v, ndraft_env, g_moe_group, int(g_qx), 1, 3, g_rp, 87, g_stream, g_static_pct_dec, g_onecopy, g_leanev, g_descstage, g_clock, 0, g_shtp});
                runs.push_back({cfg, v, ndraft_env, g_moe_group, int(g_qx), 1, 5, g_rp, 87, g_stream, g_static_pct_dec, g_onecopy, g_leanev, g_descstage, g_clock, 0, g_shtp});
                runs.push_back({cfg, v, ndraft_env, g_moe_group, int(g_qx), 1, 5, g_rp, 60, g_stream, g_static_pct_dec, g_onecopy, g_leanev, g_descstage, g_clock, 0, g_shtp});
                runs.push_back({cfg, v, ndraft_env, g_moe_group, int(g_qx), 1, 5, g_rp, 35, g_stream, g_static_pct_dec, g_onecopy, g_leanev, g_descstage, g_clock, 0, g_shtp});
                runs.push_back({cfg, v, ndraft_env, g_moe_group, int(g_qx), 1, 3, g_rp, 60, g_stream, g_static_pct_dec, g_onecopy, g_leanev, g_descstage, g_clock, 0, g_shtp});
            } else if (const char* dsw = getenv("IE_GLM52_DEC_SWEEP")) {
                // value = comma-separated dec levels, e.g. "3,4"; "1" sweeps 0,3,2
                if (std::string(dsw) == "1") {
                    runs.push_back({cfg, v, ndraft_env, g_moe_group, int(g_qx), g_hits_first, 0, g_rp, g_static_pct, g_stream, g_static_pct_dec, g_onecopy, g_leanev, g_descstage, g_clock, 0, g_shtp});
                    runs.push_back({cfg, v, ndraft_env, g_moe_group, int(g_qx), g_hits_first, 3, g_rp, g_static_pct, g_stream, g_static_pct_dec, g_onecopy, g_leanev, g_descstage, g_clock, 0, g_shtp});
                    runs.push_back({cfg, v, ndraft_env, g_moe_group, int(g_qx), g_hits_first, 2, g_rp, g_static_pct, g_stream, g_static_pct_dec, g_onecopy, g_leanev, g_descstage, g_clock, 0, g_shtp});
                } else {
                    int lv = 0;
                    for (const char* c2 = dsw; *c2; ++c2)
                        if (*c2 >= '0' && *c2 <= '9') {
                            lv = *c2 - '0';
                            runs.push_back({cfg, v, ndraft_env, g_moe_group, int(g_qx), g_hits_first, lv, g_rp, g_static_pct, g_stream, g_static_pct_dec, g_onecopy, g_leanev, g_descstage, g_clock, 0, g_shtp});
                        }
                }
            } else if (getenv("IE_GLM52_MTP_SWEEP")) {
                runs.push_back({cfg, v, 0, g_moe_group, 1, g_hits_first, g_dec_lvl, g_rp, g_static_pct, g_stream, g_static_pct_dec, g_onecopy, g_leanev, g_descstage, g_clock, 0, g_shtp}); runs.push_back({cfg, v, 1, g_moe_group, 1, g_hits_first, g_dec_lvl, g_rp, g_static_pct, g_stream, g_static_pct_dec, g_onecopy, g_leanev, g_descstage, g_clock, 0, g_shtp});
                runs.push_back({cfg, v, 2, g_moe_group, 1, g_hits_first, g_dec_lvl, g_rp, g_static_pct, g_stream, g_static_pct_dec, g_onecopy, g_leanev, g_descstage, g_clock, 0, g_shtp}); runs.push_back({cfg, v, 3, g_moe_group, 1, g_hits_first, g_dec_lvl, g_rp, g_static_pct, g_stream, g_static_pct_dec, g_onecopy, g_leanev, g_descstage, g_clock, 0, g_shtp});
            } else if (getenv("IE_GLM52_MOE_GROUP_SWEEP")) {
                // The MoE phase is 10.7 s against a ~7.7 s kernel estimate; group
                // size sets how finely expert DMA interleaves with the GEMMs, so
                // sweeping it from one load says where the gap actually is.
                for (int gsz : {16, 32, 64, 128, 256})
                    runs.push_back({cfg, v, ndraft_env, gsz, 1, g_hits_first, g_dec_lvl, g_rp, g_static_pct, g_stream, g_static_pct_dec, g_onecopy, g_leanev, g_descstage, g_clock, 0, g_shtp});
            } else if (getenv("IE_GLM52_QX_SWEEP")) {
                runs.push_back({cfg, v, ndraft_env, g_moe_group, 1, g_hits_first, g_dec_lvl, g_rp, g_static_pct, g_stream, g_static_pct_dec, g_onecopy, g_leanev, g_descstage, g_clock, 0, g_shtp});
                runs.push_back({cfg, v, ndraft_env, g_moe_group, 2, g_hits_first, g_dec_lvl, g_rp, g_static_pct, g_stream, g_static_pct_dec, g_onecopy, g_leanev, g_descstage, g_clock, 0, g_shtp});
                runs.push_back({cfg, v, ndraft_env, g_moe_group, 4, g_hits_first, g_dec_lvl, g_rp, g_static_pct, g_stream, g_static_pct_dec, g_onecopy, g_leanev, g_descstage, g_clock, 0, g_shtp});
            } else runs.push_back({cfg, v, ndraft_env, g_moe_group, int(g_qx), g_hits_first, g_dec_lvl, g_rp, g_static_pct, g_stream, g_static_pct_dec, g_onecopy, g_leanev, g_descstage, g_clock, 0, g_shtp});
        }
    }
    for (const auto& run : runs) {
        const int cfg = run[0];
        g_gemm_v2 = run[1];
        const int NDRAFT = run[2];
        if (run[3] & (1<<16)) {           // MG_SWEEP: field is MISS_GROUP
            char mgbuf[16]; snprintf(mgbuf, sizeof mgbuf, "%d", run[3] & 0xFFFF);
            setenv("IE_GLM52_MISS_GROUP", mgbuf, 1);
        } else g_moe_group = run[3];
        g_qx = size_t(run[4]);
        g_hits_first = run[5];
        g_dec_lvl = run[6]; g_dec2 = g_dec_lvl >= 2; g_dsplit = g_dec_lvl == 2;
        g_rp = run[7];
        g_static_pct = run[8];
        g_stream = run[9];
        g_static_pct_dec = run[10];
        g_onecopy = run[11];
        g_leanev  = run[12];
        g_descstage = run[13];
        g_clock = run[14];
        const int pp_tokens = run[15];
        g_shtp = run[16];
        if (g_stream) g_dsplit = 0;
        ng_use = cfg;
        // Reset so each configuration starts from the same cold state: same
        // empty caches, same empty KV, same phase counters.
        for (int d = 0; d < NG; ++d) {
            auto& c = CA[d];
            // Restore the static placement if a previous config released it to
            // the LRU and it got overwritten, so every config starts identical.
            size_t restored = 0;
            for (auto& [sidx, key] : static_map[size_t(d)]) {
                if (c.key[size_t(sidx)] == key) { c.age[size_t(sidx)] = UINT64_MAX; continue; }
                uint8_t* src = hot_ptr.empty() ? nullptr : hot_ptr[size_t(key)];
                if (!src) continue;                       // not in tier; leave as is
                if (c.key[size_t(sidx)] >= 0) c.lookup[size_t(c.key[size_t(sidx)])] = -1;
                c.qx.memcpy(c.slot[size_t(sidx)], src, slot_bytes);
                c.key[size_t(sidx)] = key;
                c.lookup[size_t(key)] = sidx;
                c.age[size_t(sidx)] = UINT64_MAX;
                ++restored;
            }
            if (restored) { c.qx.wait();
                            std::printf("gpu%d restored %zu static experts\n", d, restored); }
            // A config may change the static/demand split. Slots that WERE
            // static keep their (resident, correct) contents and become
            // evictable — dropping their keys would re-fetch data already in
            // VRAM — so only age is reset, which is what makes them LRU
            // candidates.
            const size_t newcap = (c.slot.size() * size_t(g_static_pct)) / 100;
            if (newcap < c.static_cap)
                for (size_t i = newcap; i < c.static_cap; ++i) c.age[i] = 0;
            c.static_cap = newcap;
            // Reset ONLY the demand reservoir. Clearing the whole lookup wiped
            // the statically-placed experts and dropped gpu0 to a 0% hit rate.
            for (size_t i = c.static_cap; i < c.slot.size(); ++i) {
                if (c.key[i] >= 0) c.lookup[size_t(c.key[i])] = -1;
                c.key[i] = -1;
                c.age[i] = 0;
            }
            c.hits = c.misses = c.h2d = c.pin_hits = 0;
            c.pf_issued = c.pf_used = 0;
            c.nk_total = c.nk_blocks = 0;
        }
        clk = 0;
        g_probe_chunks = 0; g_probe_calls = 0;   // re-arm the probe per config
        hp_attn=hp_route=hp_resolve=hp_bounce=hp_shared=hp_launch=hp_pf=hp_fold=0; hp_tokens=0;
        hp_r_cpy=hp_l_dep1=hp_l_rest=0; hp_bytes=0;
        t_attn = t_router = t_exp = t_shexp = t_head = t_fetch = 0;
        p_embed = p_attn = p_route = p_moe = p_dense = p_head = p_fetch = 0;
        out_ids.clear();

        std::printf("\n===== %d GPU%s | GEMM v%d | draft %d | group %d | dma %s | %s | dec %s =====\n",
                    cfg, cfg > 1 ? "s" : "", g_gemm_v2 ? 2 : 1, NDRAFT, g_moe_group,
                    g_qx == 1 ? "1 queue" : (g_qx == 2 ? "2 queues" : "4 queues"),
                    g_hits_first ? "hits-first" : "old order",
                    g_dec_lvl == 2 ? "v2-split" : (g_dec_lvl == 3 ? "v3-sync" :
                    (g_dec_lvl == 4 ? "v5-qc-copies" :
                    (g_dec_lvl == 5 ? "v6-no-events" : "v1"))));
        std::printf("        (miss path %s | static %d%% | miss group %s | events %s)\n",
                    g_stream ? "STREAM" : (g_onecopy ? "copy-1x" : "copy-2x"), g_static_pct,
                    getenv("IE_GLM52_MISS_GROUP") ? getenv("IE_GLM52_MISS_GROUP") : "64",
                    g_leanev ? "lean" : "full");
        std::printf("        (descriptors %s | eviction %s | shared-expert %s | group %d)\n",
                    g_descstage ? "STAGED" : "shared-USM", g_clock ? "CLOCK" : "linear-LRU",
                    g_shtp ? "SPLIT" : "gpu0", g_moe_group);
        auto t_pp = Clock::now();
        int pos = 0;
        if (getenv("IE_GLM52_NO_BATCH")) {
            for (size_t i = 0; i < prompt.size(); ++i, ++pos) forward(prompt[i], pos);
        } else {
            if (pp_tokens > 0 && pp_tokens < int(prompt.size())) {
                std::vector<int> sub(prompt.begin(), prompt.begin() + pp_tokens);
                forward_prefill(sub);
                pos = pp_tokens;
            } else {
            forward_prefill(prompt);          // whole prompt in one pass
            pos = int(prompt.size());
            }
        }
        const double pp_s = dsec(t_pp);

        if (g_static_pct_dec >= 0) {
            for (int d = 0; d < ng_use; ++d) {
                auto& c = CA[d];
                const size_t nc = (c.slot.size() * size_t(g_static_pct_dec)) / 100;
                if (nc < c.static_cap) {
                    // keep contents and keys — only make them evictable
                    for (size_t i = nc; i < c.static_cap; ++i)
                        if (c.age[i] == UINT64_MAX) c.age[i] = ++clk;
                    c.static_cap = nc;
                }
            }
            std::printf("decode cache split: static %d%% -> %d%% (gpu0 %zu, gpu1 %zu demand slots)\n",
                        g_static_pct, g_static_pct_dec,
                        CA[0].slot.size() - CA[0].static_cap,
                        NG > 1 ? CA[1].slot.size() - CA[1].static_cap : 0);
        }
        spin_wait(q.memcpy(hl.data(), logits, size_t(VOC)*4));
        int next = int(std::max_element(hl.begin(), hl.end()) - hl.begin());

        auto t_tg = Clock::now();
        uint64_t spec_rounds = 0, spec_accepted = 0, spec_drafted = 0;
        // Per-draft-position acceptance. Draft 1 is the only one grounded in the
        // main model's hidden state; 2+ chain MTP's own output, which the single
        // NextN block was not trained for. Splitting the rate by position says
        // whether chaining is the problem.
        std::vector<uint64_t> acc_at(64, 0), try_at(64, 0);
        if (want_mtp && NDRAFT > 0) {
            // Speculative decode with the model's own NextN block.
            //
            // The MTP block is ONE layer of 78, so a draft costs ~1.2% of a target
            // pass. Verification runs the drafts as a single batched forward,
            // which reads the resident weights once for the whole window instead
            // of once per token. Greedy accept: a draft survives only if the
            // target model's own argmax at that position agrees, so the emitted
            // sequence is EXACTLY what non-speculative greedy decode produces.
            std::vector<float> vl(static_cast<size_t>(VOC), 0.f);
            while (int(out_ids.size()) < n_pred) {
                std::vector<int> win;                 // [accepted, draft1..draftK]
                win.push_back(next);
                const float* hid = x;                 // hidden that produced `next`
                for (int d = 0; d < NDRAFT; ++d) {
                    mtp_step(hid, win.back(), pos + d);
                    spin_wait(q.memcpy(hl.data(), logits, size_t(VOC)*4));
                    win.push_back(int(std::max_element(hl.begin(), hl.end()) - hl.begin()));
                    hid = mtp_cur;                    // chain: MTP's own hidden
                }
                spec_rounds++; spec_drafted += uint64_t(NDRAFT);

                // one batched target pass over the whole window
                forward_chunk(win, pos, /*all_logits=*/true);

                int emitted = 0;
                for (int i = 0; i < int(win.size()) && int(out_ids.size()) < n_pred; ++i) {
                    out_ids.push_back(win[i]);
                    ++emitted;
                    q.memcpy(vl.data(), Tlog + int64_t(i)*VOC, size_t(VOC)*4).wait();
                    const int m = int(std::max_element(vl.begin(), vl.end()) - vl.begin());
                    if (i + 1 < int(win.size())) {
                        ++try_at[size_t(i)];
                        if (m == win[i+1]) { ++spec_accepted; ++acc_at[size_t(i)]; continue; }
                    }
                    next = m;                          // first divergence (or window end)
                    break;
                }
                pos += emitted;                        // rejected drafts are simply
                                                       // overwritten in the KV cache
                // hidden for `next` is row (emitted-1) of this pass
                q.memcpy(x, Tx + int64_t(emitted-1)*H, size_t(H)*4).wait();
            }
        } else {
            for (int i = 0; i < n_pred; ++i) {
                out_ids.push_back(next);
                forward(next, pos++);
                spin_wait(q.memcpy(hl.data(), logits, size_t(VOC)*4));
                next = int(std::max_element(hl.begin(), hl.end()) - hl.begin());
            }
        }
        const double tg_s = dsec(t_tg);

        std::printf("--- generated ---\n");
        for (int id : out_ids) {
            std::string str(toks[size_t(id)]);
            std::string o;
            for (size_t i2 = 0; i2 < str.size(); ++i2) {   // GPT-2 byte-level unescape
                if (str.compare(i2, 2, "\xc4\xa0") == 0) { o += ' '; ++i2; }
                else if (str.compare(i2, 2, "\xc4\x8a") == 0) { o += '\n'; ++i2; }
                else o += str[i2];
            }
            std::printf("%s", o.c_str());
        }
        std::printf("\n-----------------\n");

        const size_t ppn = (pp_tokens > 0 && pp_tokens < int(prompt.size()))
                           ? size_t(pp_tokens) : prompt.size();
        std::printf("prompt eval : %8.2f ms / %2zu tokens (%7.2f ms/tok, %6.2f tok/s)\n",
                    pp_s*1e3, ppn, pp_s*1e3/double(ppn), double(ppn)/pp_s);
        std::printf("decode      : %8.2f ms / %2d tokens (%7.2f ms/tok, %6.2f tok/s)\n",
                    tg_s*1e3, n_pred, tg_s*1e3/n_pred, n_pred/tg_s);
        if (spec_rounds)
            std::printf("            : MTP spec — %llu target passes, %llu/%llu drafts accepted "
                        "(%.1f%%), %.2f tokens per target pass\n",
                        (unsigned long long)spec_rounds,
                        (unsigned long long)spec_accepted, (unsigned long long)spec_drafted,
                        100.0*double(spec_accepted)/double(spec_drafted ? spec_drafted : 1),
                        double(out_ids.size())/double(spec_rounds));
        if (spec_rounds) {
            std::printf("            : acceptance by draft position:");
            for (int i = 0; i < NDRAFT; ++i)
                std::printf("  d%d %.0f%%(%llu/%llu)", i+1,
                            try_at[size_t(i)] ? 100.0*double(acc_at[size_t(i)])/double(try_at[size_t(i)]) : 0.0,
                            (unsigned long long)acc_at[size_t(i)],
                            (unsigned long long)try_at[size_t(i)]);
            std::printf("\n");
        }
        if (PROF) {
            std::printf("prefill phase totals (s):  embed %.2f  attn %.2f  router %.2f"
                        "  expert-residency %.2f  MoE GEMM %.2f  dense+shared %.2f  head %.2f\n",
                        p_embed, p_attn, p_route, p_fetch, p_moe, p_dense, p_head);
            // These timers only accumulate inside forward() -- the single-token
            // path. Prefill runs forward_chunk and is reported separately above.
            // Dividing by prompt+n_pred understated every decode phase by 1.25x.
            const double n = double(out_ids.empty() ? 1 : out_ids.size());
            std::printf("per-token phase breakdown (ms):\n"
                        "  MLA attention   %7.2f\n  router          %7.2f\n"
                        "  expert fetch    %7.2f\n  routed experts  %7.2f\n"
                        "  shared expert   %7.2f\n  lm_head         %7.2f\n",
                        t_attn/n*1e3, t_router/n*1e3, t_fetch/n*1e3,
                        t_exp/n*1e3, t_shexp/n*1e3, t_head/n*1e3);
        }
        for (int d = 0; d < cfg; ++d) {
            const auto& c = CA[d];
            std::printf("gpu%d cache: %.1f%% hit (%llu hit / %llu miss), %.2f GiB H2D | "
                        "hot tier served %.1f%% of misses | mean %.2f experts/block\n",
                        d, 100.0*double(c.hits)/double(c.hits+c.misses),
                        (unsigned long long)c.hits, (unsigned long long)c.misses,
                        c.h2d/1073741824.0,
                        c.misses ? 100.0*double(c.pin_hits)/double(c.misses) : 0.0,
                        c.nk_blocks ? double(c.nk_total)/double(c.nk_blocks) : 0.0);
            if (c.pf_issued)
                std::printf("gpu%d prefetch: %llu issued\n", d, (unsigned long long)c.pf_issued);
        }
        // IE_GLM52_DECODE2=N: after the main measurement, decode N tokens from
        // the hardcoded 12-token Fibonacci prompt (KV overwritten from pos 0).
        // Gives the short-context decode number every load, comparable across
        // configurations AND against the known-good reference continuation.
        if (const char* d2e = getenv("IE_GLM52_DECODE2")) {
            const int n2 = atoi(d2e);
            if (n2 > 0) {
                static const int fib12[12] = {7984,264,13020,729,429,4675,279,308,7563,79043,1372,13};
                std::vector<int> out2;
                int pos2 = 0;
                for (int i = 0; i < 12; ++i) forward(fib12[i], pos2++);
                // Per-phase counters. The running hp_bytes figure is cumulative
                // from the start of the config, so dividing it by decode tokens
                // charges decode for ALL of prefill's traffic — which is how a
                // "4.83 GB/token" decode figure appeared. These are deltas over
                // the decode2 window only.
                uint64_t b0 = 0, h0 = 0, m0 = 0, dr0 = 0; double ds0 = 0;
                for (int d = 0; d < ng_use; ++d)
                    { b0 += CA[d].h2d; h0 += CA[d].hits; m0 += CA[d].misses;
                      dr0 += CA[d].disk_reads; ds0 += CA[d].disk_secs; }
                // Host segment timers too: they otherwise blend this window with
                // the 2100-context decode above it, and the two have very
                // different fetch volumes.
                hp_attn=hp_route=hp_resolve=hp_bounce=hp_shared=hp_launch=hp_pf=hp_fold=0;
                hp_r_cpy=hp_l_dep1=hp_l_rest=0; hp_tokens=0;
                spin_wait(q.memcpy(hl.data(), logits, size_t(VOC)*4));
                int nx = int(std::max_element(hl.begin(), hl.end()) - hl.begin());
                const auto t2 = Clock::now();
                for (int i = 0; i < n2; ++i) {
                    out2.push_back(nx);
                    forward(nx, pos2++);
                    spin_wait(q.memcpy(hl.data(), logits, size_t(VOC)*4));
                    nx = int(std::max_element(hl.begin(), hl.end()) - hl.begin());
                }
                const double s2 = dsec(t2);
                uint64_t b1 = 0, h1 = 0, m1 = 0, dr1 = 0; double ds1 = 0;
                for (int d = 0; d < ng_use; ++d)
                    { b1 += CA[d].h2d; h1 += CA[d].hits; m1 += CA[d].misses;
                      dr1 += CA[d].disk_reads; ds1 += CA[d].disk_secs; }
                std::printf("decode2 tier misses -> DISK: %llu reads (%.2f/token), %.1f ms/token "
                            "blocking the host\n",
                            (unsigned long long)(dr1 - dr0), double(dr1 - dr0)/n2,
                            (ds1 - ds0)*1e3/n2);
                const double gbt = double(b1 - b0)/1e9/n2;
                const uint64_t hh = h1 - h0, mm = m1 - m0;
                std::printf("decode2 (12-tok prompt): %8.2f ms / %d tokens (%6.2f ms/tok, %5.2f tok/s)\n",
                            s2*1e3, n2, s2*1e3/n2, double(n2)/s2);
                if (g_hostprof && hp_tokens) {
                    const double n = double(hp_tokens);
                    const double acct = (hp_attn+hp_route+hp_resolve+hp_bounce+hp_shared
                                         +hp_launch+hp_pf+hp_fold)/n*1e3;
                    std::printf("decode2 host profile (ms/tok): attn %.2f route %.2f resolve %.2f "
                                "shared %.2f launch %.2f fold %.2f | segments %.1f of %.1f total\n",
                                hp_attn/n*1e3, hp_route/n*1e3, hp_resolve/n*1e3, hp_shared/n*1e3,
                                hp_launch/n*1e3, hp_fold/n*1e3, acct, s2*1e3/n2);
                }
                std::printf("decode2 phase only: %.2f GB/token fetched, cache %.1f%% hit "
                            "(%llu hit / %llu miss) | PCIe floor %.1f tok/s at 53.1 GB/s, "
                            "actual %.0f%% of link\n",
                            gbt, hh+mm ? 100.0*double(hh)/double(hh+mm) : 0.0,
                            (unsigned long long)hh, (unsigned long long)mm,
                            gbt > 0 ? 53.1/gbt : 0.0,
                            gbt > 0 ? (gbt/(s2/n2))/53.1*100.0 : 0.0);
                std::printf("decode2 text: ");
                for (int id : out2) {
                    std::string str(toks[size_t(id)]);
                    std::string o;
                    for (size_t i2 = 0; i2 < str.size(); ++i2) {
                        if (str.compare(i2, 2, "\xc4\xa0") == 0) { o += ' '; ++i2; }
                        else if (str.compare(i2, 2, "\xc4\x8a") == 0) { o += '\n'; ++i2; }
                        else o += str[i2];
                    }
                    std::printf("%s", o.c_str());
                }
                std::printf("\n");
            }
        }
    }
    if (trace) { fclose(trace); std::printf("router trace written to %s\n", getenv("IE_GLM52_TRACE")); }
    return 0;
}
