// tools/q35m_rows_test.cpp -- ie-q35m-rows-test: P4 B14 phase 1a (~/ds41_work/p60/b14/PLAN.md sections 3a and 4): can the
// 35B-A3B crown split (qwen35moe, Qwen35MoeSplitModel) decode G lanes' rows in ONE card step, bit-identical to G one-row steps,
// and what does such a step cost?
//
// --exact: every op a row-batched crown decode step would share between lanes, at rows M = --rows (default every M in 2..16),
//   compared ROW BY ROW, byte for byte, against the op the crown runs today for one row on that row's input (its T == 1
//   route), at the checkpoint's shapes (Qwen3.8-35B-A3B-Q8_0.gguf: hidden 2048, 256 experts top-8, expert and shared-expert
//   FFN 512, q+gate 8192, kv 512, DeltaNet qkv 8192 / inner 4096, vocab 248,320, alpha/beta 2048 x 32 -> F16 padded to 64):
//     * rms_norm_f32w and residual_add_rms_norm_fused (the attention and post-attention norms);
//     * quantize_q8_1 over M rows (the block_q8_1x stream: row-independent per the code's comment);
//     * the dense int-dot GEMVs forced the way the 27B's forward_slots forces them (quantize_q8_1 over M x K +
//       gemv_q8_0_soa_q8_batched) against the crown's T == 1 route (quantize_q8_1 + gemv_q8_0_soa_q8), at every dense shape;
//     * the MoE Q8_0 decode chain, stage by stage: moe_router (topk ids and weights), quantize_q8_1, moe_gate_up_silu_q8,
//       quantize_q8_1 of h, moe_down_q8, moe_reduce_q8 -- each batched stage fed the batched previous stage's output;
//     * the head: rms_norm + the batched int-dot GEMV over the vocab (and how many rows' argmax changed, if any);
//     * (P4 B14 phase 1b) the row-wise ops the rows path runs once over the G rows: rms_norm_f32w at the head width over
//       M x heads rows (the q / k norms), rope_partial with a DIFFERENT position in every row, embedding_lookup_q8_0 over M
//       ids, split_q_gate_per_head, sigmoid_gate, cast_qkv_split_fp16_to_fp32, l2_norm_scale, repeat_interleave_heads,
//       extract_cols, compute_g_beta_h16, gated_rms_norm, swiglu, residual_add and the shared-expert gate (the crown's own
//       lambda, copied verbatim);
//     * REPORTED, never a FAIL: the alpha/beta projections through dense::gemv_q_T (F16: a GEMV at T == 1, a GEMM route at
//       T >= 2), which a row step can run per row (N = 64: cheap).
//   A difference in a must-match op is a FAIL (exit 1).
// --timing: the shared-weight part of a card's decode step at G = 1..16 rows on --layers distinct layers (default 20 = one
//   card of the split: 15 DeltaNet + 5 full attention, every 4th full, as the checkpoint's full_attention_interval 4) plus the
//   head (--no-head drops it): per layer the norms, every dense projection (G == 1 the T == 1 kernels, G >= 2 the batched
//   int-dot), alpha/beta per row, the router, the Q8_0 MoE decode chain over G x 8 slots and the shared expert. NOT timed:
//   the per-lane parts (causal conv, DeltaNet recurrence, q/k norms, RoPE, fa2 decode attention) -- they stay per lane in the
//   design and add ~linearly; B10's step traces carry them. Each layer restarts from a fixed residual (a 64 KB copy: the
//   random weights would otherwise overflow fp16 across layers). Median of --reps steps per G (after 3 warm-ups): ms,
//   t(G)/t(1), the marginal ms per row, the distinct experts G rows touch (layer 0) and the effective GB/s over the bytes a
//   step reads. The kill criterion of phase 1a is printed: t(4)/t(1) > 3.
// --dry prints the shapes, the rows and the planned VRAM, then exits WITHOUT touching a device; --selftest runs the host-side
// checks (the byte comparator, the percentile, the expert-coverage formula, the SoA strides) without a device. Default:
// --exact then --timing on --device 0 (ZE_AFFINITY_MASK / ONEAPI_DEVICE_SELECTOR narrow it further).
//   usage: ie-q35m-rows-test [--dry] [--selftest] [--exact] [--timing] [--rows 2,3,...,16] [--layers 20] [--reps 10]
//                            [--no-head] [--device 0]
#include "ie/moe_q8.hpp"
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"
#include "../src/model/dense_dispatch.hpp"   // dense::gemv_q_T, residual_add_rms_norm_fused (the crown's own routes)

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

using namespace ie;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what, const std::string& detail = "") {
    std::printf("[%s] %s%s%s\n", ok ? " ok " : "FAIL", what.c_str(), detail.empty() ? "" : "  ", detail.c_str());
    std::fflush(stdout);
    if (!ok) ++g_fail;
}
void info(const std::string& what, const std::string& detail) { std::printf("[info] %s  %s\n", what.c_str(), detail.c_str()); std::fflush(stdout); }

// the checkpoint's shapes (verified from the GGUF header of Qwen3.8-35B-A3B-Q8_0.gguf, 2026-09-27)
constexpr uint32_t H = 2048, E = 256, TK = 8, EFF = 512, SH_FF = 512;
constexpr uint32_t N_QG = 8192, N_Q = 4096, N_KV = 512, CONV_CH = 8192, SI = 4096, AB_N = 64, V = 248320;
constexpr uint32_t kMaxRows = 16;   // the batched int-dot leaves take 2..16 rows (ops.hpp gemv_q8_0_soa_q8_batched)
constexpr float kEps = 1e-6f;

struct Dense { uint32_t K, N; const char* name; };
const Dense kDense[] = {
    {H, CONV_CH, "attn_qkv (DeltaNet)"}, {H, SI, "attn_gate (DeltaNet z)"}, {SI, H, "ssm_out"},
    {H, N_QG, "attn_q (+gate)"}, {H, N_KV, "attn_k / attn_v"}, {N_Q, H, "attn_output"},
    {H, SH_FF, "shared gate / up"}, {SH_FF, H, "shared down"},
};

// Row r of `batched` (an M-row result, row stride n) against `solo` (the 1-row result on row r).
struct RowDiff { uint32_t rows_differ = 0; double max_abs = 0; uint32_t argmax_changes = 0; };
template <class T>
bool row_same(const T* a, const T* b, size_t n) { return std::memcmp(a, b, n * sizeof(T)) == 0; }
template <class T>
void row_diff(const std::vector<T>& solo, const std::vector<T>& batched, uint32_t r, size_t n, RowDiff& d, bool argmax) {
    const T* b = batched.data() + size_t(r) * n;
    if (row_same(solo.data(), b, n)) return;
    ++d.rows_differ;
    if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, sycl::half>)
        for (size_t i = 0; i < n; ++i) d.max_abs = std::max(d.max_abs, std::fabs(double(float(solo[i])) - double(float(b[i]))));
    if (argmax) {
        auto am = [&](const T* p) { size_t best = 0; for (size_t i = 1; i < n; ++i) if (float(p[i]) > float(p[best])) best = i; return best; };
        if (am(solo.data()) != am(b)) ++d.argmax_changes;
    }
}
std::string fmt(const RowDiff& d, uint32_t rows, bool argmax) {
    if (!d.rows_differ) return "bit-identical";
    char b[200];
    std::snprintf(b, sizeof b, "%u of %u rows differ, max abs %.3g%s%s", d.rows_differ, rows, d.max_abs, argmax ? ", argmax changes " : "",
                  argmax ? std::to_string(d.argmax_changes).c_str() : "");
    return b;
}

double pctl(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    const size_t i = std::min(v.size() - 1, size_t(std::lround(p / 100.0 * double(v.size() - 1))));
    return v[i];
}

// distinct experts G rows touch when each picks TK of E uniformly (the plan's 256(1 - (248/256)^G))
double expected_distinct(uint32_t G) { return double(E) * (1.0 - std::pow(double(E - TK) / double(E), double(G))); }

// the SoA Q8_0 plane sizes of a [K -> N] weight: qs int8 [N * K], d fp16 [N * K / 32]
uint64_t soa_bytes(uint32_t K, uint32_t N) { return uint64_t(N) * K + uint64_t(N) * (K / 32) * 2; }
uint64_t expert_layer_bytes() { return uint64_t(E) * (2 * soa_bytes(H, EFF) + soa_bytes(EFF, H)); }
uint64_t dense_layer_bytes(bool full) {
    uint64_t b = soa_bytes(H, SH_FF) * 2 + soa_bytes(SH_FF, H) + uint64_t(E) * H * 4;   // shared expert + router (f32)
    if (full) b += soa_bytes(H, N_QG) + 2 * soa_bytes(H, N_KV) + soa_bytes(N_Q, H);
    else b += soa_bytes(H, CONV_CH) + soa_bytes(H, SI) + soa_bytes(SI, H) + 2ull * H * AB_N * 2;
    return b;
}

std::vector<uint32_t> parse_list(const std::string& s) {
    std::vector<uint32_t> v;
    size_t i = 0;
    while (i < s.size()) {
        const size_t j = s.find(',', i);
        v.push_back(uint32_t(std::atoi(s.substr(i, j == std::string::npos ? std::string::npos : j - i).c_str())));
        if (j == std::string::npos) break;
        i = j + 1;
    }
    return v;
}

int selftest() {
    std::vector<sycl::half> solo(4, sycl::half(1.f)), batched(8, sycl::half(1.f));
    RowDiff d; row_diff(solo, batched, 0, 4, d, true); row_diff(solo, batched, 1, 4, d, true);
    check(d.rows_differ == 0 && fmt(d, 2, true) == "bit-identical", "selftest: identical rows compare equal");
    batched[5] = sycl::half(3.f);
    RowDiff e; row_diff(solo, batched, 1, 4, e, true);
    check(e.rows_differ == 1 && e.argmax_changes == 1 && e.max_abs == 2.0, "selftest: a changed element is found, with its size and the argmax move");
    std::vector<float> fz{0.f}, fnz{-0.f};   // bytes, not values: -0 != +0 is a difference (a reduction order shows up here)
    RowDiff z; row_diff(fz, fnz, 0, 1, z, false);
    check(z.rows_differ == 1, "selftest: the comparison is bytewise (+0 vs -0 differs)");
    check(pctl({5, 1, 3, 2, 4}, 50) == 3 && pctl({5, 1, 3, 2, 4}, 100) == 5 && pctl({}, 50) == 0, "selftest: percentile");
    check(std::fabs(expected_distinct(1) - 8.0) < 1e-9 && std::fabs(expected_distinct(4) - 30.5) < 0.1 &&
              std::fabs(expected_distinct(8) - 57.0) < 0.5 && std::fabs(expected_distinct(16) - 101.9) < 0.5,
          "selftest: distinct experts 8 / 30.5 / 57 / 102 at G = 1 / 4 / 8 / 16 (the plan's numbers)");
    check(soa_bytes(H, EFF) == 1048576ull + 65536ull, "selftest: SoA bytes of one 2048 -> 512 expert matrix (qs 1 MiB + d 64 KiB)");
    check(sizeof(block_q8_1x) == 40, "selftest: block_q8_1x is 40 bytes (32 int8 + scale + sum)");
    check(parse_list("2,3,5,8,12,16") == std::vector<uint32_t>({2, 3, 5, 8, 12, 16}), "selftest: the --rows list");
    return g_fail;
}

void dry(const std::vector<uint32_t>& rows, uint32_t layers, bool head) {
    std::printf("shapes: hidden %u, experts %u top-%u, expert FFN %u, shared FFN %u, q+gate %u, kv %u, DeltaNet qkv %u / inner %u, vocab %u\n",
                H, E, TK, EFF, SH_FF, N_QG, N_KV, CONV_CH, SI, V);
    std::printf("exact: rows");
    for (uint32_t m : rows) std::printf(" %u", m);
    std::printf(" against the 1-row route; dense shapes:");
    for (const Dense& d : kDense) std::printf(" %s %ux%u;", d.name, d.K, d.N);
    const uint64_t exact_vram = expert_layer_bytes() + soa_bytes(H, V) + soa_bytes(H, CONV_CH) + (64ull << 20);
    std::printf("\nexact VRAM ~%.2f GiB (one layer's experts, the head, the largest dense weight, activations)\n", exact_vram / 1073741824.0);
    const uint32_t nfull = layers / 4, nlin = layers - nfull;
    const uint64_t t_vram = uint64_t(layers) * expert_layer_bytes() + uint64_t(nlin) * dense_layer_bytes(false) +
                            uint64_t(nfull) * dense_layer_bytes(true) + (head ? soa_bytes(H, V) : 0) + (64ull << 20);
    std::printf("timing: G = 1..%u on %u layers (%u DeltaNet + %u full attention)%s: VRAM ~%.2f GiB\n", kMaxRows, layers, nlin, nfull,
                head ? " + the head" : "", t_vram / 1073741824.0);
    std::printf("expected distinct experts per layer:");
    for (uint32_t g : {1u, 2u, 4u, 8u, 16u}) std::printf(" G=%u %.1f", g, expected_distinct(g));
    std::printf("\nper step bytes at G = 1: %.2f GB (weights of the dense set + 8 experts per layer%s)\n",
                (double(nlin) * dense_layer_bytes(false) + double(nfull) * dense_layer_bytes(true) +
                 double(layers) * expected_distinct(1) * double(2 * soa_bytes(H, EFF) + soa_bytes(EFF, H)) + (head ? double(soa_bytes(H, V)) : 0.0)) / 1e9,
                head ? " + the head" : "");
}

// ---- device helpers ------------------------------------------------------------------------------------------------------
// a device-side pseudo-random fill: values in about (-a, a) (+ b)
template <class T>
void fill_dev(sycl::queue& q, T* p, size_t n, uint32_t seed, float a, float b = 0.f) {
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        uint32_t h = uint32_t(i[0]) * 2654435761u ^ (seed * 0x9E3779B9u); h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
        p[i] = T((float(h & 0xFFFFFFu) / float(0xFFFFFF) * 2.f - 1.f) * a + b);
    }).wait();
}

struct Soa { int8_t* qs = nullptr; uint16_t* d = nullptr; uint32_t K = 0, N = 0; };

struct Dev {
    sycl::queue q;
    std::vector<void*> owned;
    template <class T> T* alloc(size_t n) {
        T* p = sycl::malloc_device<T>(n, q);
        if (!p) { std::fprintf(stderr, "device allocation of %zu x %zu B failed\n", n, sizeof(T)); std::exit(2); }
        owned.push_back(p);
        return p;
    }
    Soa soa(uint32_t K, uint32_t N, uint32_t seed) {
        Soa w; w.K = K; w.N = N;
        w.qs = alloc<int8_t>(size_t(N) * K);
        auto* d16 = alloc<sycl::half>(size_t(N) * (K / 32));
        fill_dev(q, w.qs, size_t(N) * K, seed, 120.f);
        fill_dev(q, d16, size_t(N) * (K / 32), seed + 1, 0.004f, 0.006f);   // scales in (0.002, 0.01)
        w.d = reinterpret_cast<uint16_t*>(d16);
        return w;
    }
    template <class T> std::vector<T> down(const T* p, size_t n) { std::vector<T> v(n); q.memcpy(v.data(), p, n * sizeof(T)).wait(); return v; }
    ~Dev() { for (void* p : owned) sycl::free(p, q); }
};

// the crown's dense projection for G rows: T == 1 its sgemv's route, G >= 2 forward_slots' forced batched int-dot
sycl::event dense_rows(sycl::queue& q, const sycl::half* A, const Soa& w, sycl::half* out, uint32_t G, void* act) {
    quantize_q8_1(q, A, act, G * w.K);
    return G == 1 ? gemv_q8_0_soa_q8(q, act, w.qs, w.d, out, w.K, w.N) : gemv_q8_0_soa_q8_batched(q, act, w.qs, w.d, out, w.K, w.N, G);
}

struct Moe { Soa gate, up, down; float* router = nullptr; };
Moe make_moe(Dev& d, uint32_t seed) {
    Moe m;
    m.gate = d.soa(H, EFF * E, seed); m.up = d.soa(H, EFF * E, seed + 10); m.down = d.soa(EFF, H * E, seed + 20);
    m.router = d.alloc<float>(size_t(E) * H);
    fill_dev(d.q, m.router, size_t(E) * H, seed + 30, 0.05f);
    return m;
}

// the MoE decode chain buffers for up to kMaxRows rows
struct MoeWs { int32_t* idx; sycl::half* w; void* xq8; sycl::half* h; void* hq8; sycl::half* yp; sycl::half* y; };
MoeWs make_moe_ws(Dev& d) {
    MoeWs s;
    s.idx = d.alloc<int32_t>(kMaxRows * TK); s.w = d.alloc<sycl::half>(kMaxRows * TK);
    s.xq8 = d.alloc<block_q8_1x>(kMaxRows * H / 32); s.h = d.alloc<sycl::half>(size_t(kMaxRows) * TK * EFF);
    s.hq8 = d.alloc<block_q8_1x>(size_t(kMaxRows) * TK * EFF / 32); s.yp = d.alloc<sycl::half>(size_t(kMaxRows) * TK * H);
    s.y = d.alloc<sycl::half>(size_t(kMaxRows) * H);
    return s;
}
// E expert planes: gate/up [E][EFF x H] (qs stride EFF * H, d stride EFF * H / 32), down [E][H x EFF]
void moe_chain(sycl::queue& q, const Moe& m, const MoeWs& s, const sycl::half* xn, uint32_t G) {
    // the rows path routes G >= 2 rows with moe_router_rows (== the T == 1 router per row; the fused T > 1 kernel is not)
    if (G == 1) moe_router(q, xn, m.router, s.idx, s.w, 1, H, E, TK);
    else        moe_router_rows(q, xn, m.router, s.idx, s.w, G, H, E, TK);
    quantize_q8_1(q, xn, s.xq8, G * H);
    moe_gate_up_silu_q8(q, s.xq8, m.gate.qs, m.gate.d, m.up.qs, m.up.d, uint64_t(EFF) * H, uint64_t(EFF) * H / 32, s.idx, s.h, G, TK, H, EFF);
    quantize_q8_1(q, s.h, s.hq8, G * TK * EFF);
    moe_down_q8(q, s.hq8, m.down.qs, m.down.d, uint64_t(H) * EFF, uint64_t(H) * EFF / 32, s.idx, s.w, s.yp, G, TK, EFF, H);
    moe_reduce_q8(q, s.yp, s.y, G, TK, H);
}

// COPY of qwen35moe_split.cpp's extract_cols (itself a copy of qwen35_split.cpp's).
sycl::event extract_cols(sycl::queue& q, const sycl::half* src, sycl::half* dst, uint32_t T, uint32_t nh, uint32_t src_stride) {
    return q.parallel_for(sycl::range<1>(uint64_t(T) * nh), [=](sycl::id<1> i) {
        const uint32_t t = uint32_t(i) / nh, h = uint32_t(i) % nh;
        dst[uint64_t(t) * nh + h] = src[uint64_t(t) * src_stride + h];
    });
}
// COPY of the crown's shared-expert gate (qwen35moe_split.cpp run_layers / run_layers_rows): my += eo * sigmoid(gw . xn)
void shexp_gate(sycl::queue& q, const float* gw, const sycl::half* xn, const sycl::half* eo, sycl::half* my, float* dng, uint32_t T) {
    const uint32_t HH = H;
    q.parallel_for(sycl::range<1>(T), [=](sycl::id<1> ti) {
        const uint32_t t = uint32_t(ti); float gg = 0.f;
        for (uint32_t h = 0; h < HH; ++h) gg += gw[h] * float(xn[uint64_t(t) * HH + h]);
        dng[t] = 1.f / (1.f + sycl::exp(-gg));
    });
    q.parallel_for(sycl::range<1>(uint64_t(T) * HH), [=](sycl::id<1> i) {
        const uint32_t t = uint32_t(uint64_t(i) / HH);
        my[i] = sycl::half(float(my[i]) + float(eo[i]) * dng[t]);
    });
}

// An op's M-row call against M one-row calls: solo(r) = row r's output bytes (one-row call on row r's input), batch(M) = the
// M-row call's output bytes laid out row by row (row_bytes each).
template <class Solo, class Batch>
void rows_op(const std::vector<uint32_t>& rows, const std::string& name, size_t row_bytes, Solo solo, Batch batch) {
    std::vector<std::vector<uint8_t>> s(kMaxRows);
    for (uint32_t r = 0; r < kMaxRows; ++r) s[r] = solo(r);
    for (uint32_t M : rows) {
        const std::vector<uint8_t> b = batch(M);
        RowDiff dd;
        for (uint32_t r = 0; r < M; ++r) row_diff(s[r], b, r, row_bytes, dd, false);
        check(!dd.rows_differ && b.size() == size_t(M) * row_bytes, name + " M = " + std::to_string(M), fmt(dd, M, false));
    }
}

// the row-wise ops of the rows path (P4 B14 phase 1b), each against its one-row call
void exact_rowwise(Dev& d, const std::vector<uint32_t>& rows) {
    sycl::queue& q = d.q;
    constexpr uint32_t HD = 256, NQH = N_Q / HD, NKH = N_KV / HD, ROPE_N = 64;
    constexpr uint32_t SKH = 16, SVH = 32, SHD = 128, KW = SKH * SHD, EMB_V = 1024;
    constexpr float kTheta = 1e7f;
    auto* xa = d.alloc<sycl::half>(size_t(kMaxRows) * N_QG);      // fp16 inputs (the widest row: 8192)
    auto* xb = d.alloc<sycl::half>(size_t(kMaxRows) * N_QG);
    auto* fa = d.alloc<float>(size_t(kMaxRows) * SI);             // fp32 inputs
    fill_dev(q, xa, size_t(kMaxRows) * N_QG, 41, 1.f);
    fill_dev(q, xb, size_t(kMaxRows) * N_QG, 42, 1.f);
    fill_dev(q, fa, size_t(kMaxRows) * SI, 43, 1.f);
    auto* oh = d.alloc<sycl::half>(size_t(kMaxRows) * N_QG);
    auto* oh2 = d.alloc<sycl::half>(size_t(kMaxRows) * N_QG);
    auto* of = d.alloc<float>(size_t(kMaxRows) * SI);
    auto* of2 = d.alloc<float>(size_t(kMaxRows) * SI);
    auto* of3 = d.alloc<float>(size_t(kMaxRows) * SI);
    auto* whd = d.alloc<float>(HD); fill_dev(q, whd, HD, 44, 0.2f, 1.f);
    auto* wsh = d.alloc<sycl::half>(SHD); fill_dev(q, wsh, SHD, 45, 0.2f, 1.f);
    auto* alog = d.alloc<float>(SVH); fill_dev(q, alog, SVH, 46, 0.5f);
    auto* dtb = d.alloc<float>(SVH); fill_dev(q, dtb, SVH, 47, 0.5f);
    auto* gw = d.alloc<float>(H); fill_dev(q, gw, H, 48, 0.05f);
    auto* dng = d.alloc<float>(kMaxRows);
    auto* pos = d.alloc<int32_t>(kMaxRows);
    auto* ids = d.alloc<int32_t>(kMaxRows);
    {
        std::vector<int32_t> hp(kMaxRows), hi(kMaxRows);
        for (uint32_t r = 0; r < kMaxRows; ++r) { hp[r] = int32_t(37 + 1013 * r); hi[r] = int32_t((r * 389 + 5) % EMB_V); }
        q.memcpy(pos, hp.data(), kMaxRows * 4); q.memcpy(ids, hi.data(), kMaxRows * 4).wait();
    }
    const size_t emb_bytes = size_t(EMB_V) * (H / 32) * sizeof(block_q8_0);
    auto* emb = d.alloc<uint8_t>(emb_bytes);
    {   // a Q8_0 table: random int8 quants, finite fp16 scales
        std::vector<block_q8_0> t(size_t(EMB_V) * (H / 32));
        uint32_t st = 12345;
        for (auto& b : t) {
            st = st * 1664525u + 1013904223u;
            b.d = sycl::bit_cast<uint16_t>(sycl::half(0.002f + float(st >> 8 & 0xFFFF) / 65535.f * 0.01f));
            for (auto& v : b.qs) { st = st * 1664525u + 1013904223u; v = int8_t(st >> 24); }
        }
        q.memcpy(emb, t.data(), emb_bytes).wait();
    }
    auto bytes = [&](const void* p, size_t n) { return d.down(static_cast<const uint8_t*>(p), n); };
    // rows gathered from several outputs: row r = the concatenation of out[k] + r * stride[k], len[k] bytes
    auto gather = [&](uint32_t M, std::vector<std::tuple<const void*, size_t>> outs) {
        std::vector<std::vector<uint8_t>> v;
        size_t row = 0;
        for (auto& [p, len] : outs) { v.push_back(bytes(p, size_t(M) * len)); row += len; }
        std::vector<uint8_t> o;
        o.reserve(size_t(M) * row);
        for (uint32_t r = 0; r < M; ++r)
            for (size_t k = 0; k < outs.size(); ++k) {
                const size_t len = std::get<1>(outs[k]);
                o.insert(o.end(), v[k].begin() + size_t(r) * len, v[k].begin() + size_t(r + 1) * len);
            }
        return o;
    };
    const size_t h2 = sizeof(sycl::half), f4 = sizeof(float);

    for (const auto& hn : {std::pair<uint32_t, const char*>{NQH, "q"}, {NKH, "k"}}) {
        const uint32_t nh = hn.first;   // (plain locals: the lambdas below capture them)
        const char* nm = hn.second;
        const uint32_t W = nh * HD;
        rows_op(rows, std::string("rms_norm_f32w at the head width (") + nm + " norm, " + std::to_string(nh) + " heads x 256)", W * h2,
                [&](uint32_t r) { rms_norm_f32w(q, xa + size_t(r) * W, whd, oh, nh, HD, kEps).wait(); return bytes(oh, W * h2); },
                [&](uint32_t M) { rms_norm_f32w(q, xa, whd, oh, M * nh, HD, kEps).wait(); return bytes(oh, size_t(M) * W * h2); });
        rows_op(rows, std::string("rope_partial (") + nm + ", a different position in every row)", W * h2,
                [&](uint32_t r) { rope_partial(q, xa + size_t(r) * W, pos + r, oh, 1, nh, HD, ROPE_N, kTheta).wait(); return bytes(oh, W * h2); },
                [&](uint32_t M) { rope_partial(q, xa, pos, oh, M, nh, HD, ROPE_N, kTheta).wait(); return bytes(oh, size_t(M) * W * h2); });
    }
    rows_op(rows, "embedding_lookup_q8_0 over M ids", H * h2,
            [&](uint32_t r) { embedding_lookup_q8_0(q, ids + r, emb, oh, 1, H).wait(); return bytes(oh, H * h2); },
            [&](uint32_t M) { embedding_lookup_q8_0(q, ids, emb, oh, M, H).wait(); return bytes(oh, size_t(M) * H * h2); });
    rows_op(rows, "split_q_gate_per_head (q and gate)", 2 * N_Q * h2,
            [&](uint32_t r) { split_q_gate_per_head(q, xa + size_t(r) * N_QG, oh, oh2, 1, NQH, HD).wait(); return gather(1, {{oh, N_Q * h2}, {oh2, N_Q * h2}}); },
            [&](uint32_t M) { split_q_gate_per_head(q, xa, oh, oh2, M, NQH, HD).wait(); return gather(M, {{oh, N_Q * h2}, {oh2, N_Q * h2}}); });
    rows_op(rows, "sigmoid_gate", N_Q * h2,
            [&](uint32_t r) { sigmoid_gate(q, xa + size_t(r) * N_Q, xb + size_t(r) * N_Q, oh, N_Q).wait(); return bytes(oh, N_Q * h2); },
            [&](uint32_t M) { sigmoid_gate(q, xa, xb, oh, uint64_t(M) * N_Q).wait(); return bytes(oh, size_t(M) * N_Q * h2); });
    rows_op(rows, "cast_qkv_split_fp16_to_fp32 (q, k, v)", (2 * KW + SI) * f4,
            [&](uint32_t r) { cast_qkv_split_fp16_to_fp32(q, xa + size_t(r) * CONV_CH, of, of2, of3, 1, KW, SI).wait();
                              return gather(1, {{of, KW * f4}, {of2, KW * f4}, {of3, SI * f4}}); },
            [&](uint32_t M) { cast_qkv_split_fp16_to_fp32(q, xa, of, of2, of3, M, KW, SI).wait();
                              return gather(M, {{of, KW * f4}, {of2, KW * f4}, {of3, SI * f4}}); });
    rows_op(rows, "l2_norm_scale (16 heads x 128 a row, the q scale)", KW * f4,
            [&](uint32_t r) { l2_norm_scale(q, fa + size_t(r) * KW, of, SKH, SHD, 0.0883883f, 1e-6f).wait(); return bytes(of, KW * f4); },
            [&](uint32_t M) { l2_norm_scale(q, fa, of, M * SKH, SHD, 0.0883883f, 1e-6f).wait(); return bytes(of, size_t(M) * KW * f4); });
    rows_op(rows, "repeat_interleave_heads (16 -> 32 heads, TILE)", SI * f4,
            [&](uint32_t r) { repeat_interleave_heads(q, fa + size_t(r) * KW, of, 1, SKH, SHD, SVH / SKH).wait(); return bytes(of, SI * f4); },
            [&](uint32_t M) { repeat_interleave_heads(q, fa, of, M, SKH, SHD, SVH / SKH).wait(); return bytes(of, size_t(M) * SI * f4); });
    rows_op(rows, "extract_cols (alpha 64 -> 32)", SVH * h2,
            [&](uint32_t r) { extract_cols(q, xa + size_t(r) * AB_N, oh, 1, SVH, AB_N).wait(); return bytes(oh, SVH * h2); },
            [&](uint32_t M) { extract_cols(q, xa, oh, M, SVH, AB_N).wait(); return bytes(oh, size_t(M) * SVH * h2); });
    rows_op(rows, "compute_g_beta_h16 (g and beta)", 2 * SVH * f4,
            [&](uint32_t r) { compute_g_beta_h16(q, xa + size_t(r) * SVH, xb + size_t(r) * SVH, alog, dtb, of, of2, 1, SVH).wait();
                              return gather(1, {{of, SVH * f4}, {of2, SVH * f4}}); },
            [&](uint32_t M) { compute_g_beta_h16(q, xa, xb, alog, dtb, of, of2, M, SVH).wait(); return gather(M, {{of, SVH * f4}, {of2, SVH * f4}}); });
    rows_op(rows, "gated_rms_norm (32 heads x 128 a row)", SI * h2,
            [&](uint32_t r) { gated_rms_norm(q, fa + size_t(r) * SI, xa + size_t(r) * SI, wsh, oh, SVH, SHD, kEps).wait(); return bytes(oh, SI * h2); },
            [&](uint32_t M) { gated_rms_norm(q, fa, xa, wsh, oh, M * SVH, SHD, kEps).wait(); return bytes(oh, size_t(M) * SI * h2); });
    rows_op(rows, "swiglu (shared expert 512)", SH_FF * h2,
            [&](uint32_t r) { swiglu(q, xa + size_t(r) * SH_FF, xb + size_t(r) * SH_FF, oh, SH_FF).wait(); return bytes(oh, SH_FF * h2); },
            [&](uint32_t M) { swiglu(q, xa, xb, oh, uint64_t(M) * SH_FF).wait(); return bytes(oh, size_t(M) * SH_FF * h2); });
    rows_op(rows, "residual_add (hidden)", H * h2,
            [&](uint32_t r) { residual_add(q, xa + size_t(r) * H, xb + size_t(r) * H, oh, H).wait(); return bytes(oh, H * h2); },
            [&](uint32_t M) { residual_add(q, xa, xb, oh, uint64_t(M) * H).wait(); return bytes(oh, size_t(M) * H * h2); });
    rows_op(rows, "the shared-expert gate (my += eo * sigmoid(gw . xn), the crown's lambda)", H * h2,
            [&](uint32_t r) { q.memcpy(oh, xb + size_t(r) * H, H * h2); shexp_gate(q, gw, xa + size_t(r) * H, xa + size_t(8 + r) * H, oh, dng, 1); q.wait();
                              return bytes(oh, H * h2); },
            [&](uint32_t M) { q.memcpy(oh, xb, size_t(M) * H * h2);
                              // eo rows 8 + r: the same rows the solo calls read (xa has kMaxRows x 8192 halves: room for 8 + 16 rows of H)
                              shexp_gate(q, gw, xa, xa + size_t(8) * H, oh, dng, M); q.wait();
                              return bytes(oh, size_t(M) * H * h2); });
}

// ---- --exact -------------------------------------------------------------------------------------------------------------
void exact(Dev& d, const std::vector<uint32_t>& rows) {
    sycl::queue& q = d.q;
    auto* x = d.alloc<sycl::half>(size_t(kMaxRows) * SI);     // the input rows (the widest K: 4096)
    auto* x2 = d.alloc<sycl::half>(size_t(kMaxRows) * H);     // a second input (the residual's block)
    fill_dev(q, x, size_t(kMaxRows) * SI, 1, 1.f);
    fill_dev(q, x2, size_t(kMaxRows) * H, 2, 1.f);
    auto* nw = d.alloc<float>(H);
    fill_dev(q, nw, H, 3, 0.2f, 1.f);
    auto* y = d.alloc<sycl::half>(size_t(kMaxRows) * V);
    auto* xr = d.alloc<sycl::half>(size_t(kMaxRows) * H);     // residual scratch (fused norm writes it)
    auto* act = d.alloc<block_q8_1x>(size_t(kMaxRows) * SI / 32);
    exact_rowwise(d, rows);

    // norms: rms_norm_f32w, residual_add_rms_norm_fused (x updated in place: both outputs compared)
    {
        std::vector<std::vector<sycl::half>> s(kMaxRows), sr(kMaxRows), sx(kMaxRows);
        for (uint32_t r = 0; r < kMaxRows; ++r) {
            rms_norm_f32w(q, x + size_t(r) * H, nw, y, 1, H, kEps).wait();
            s[r] = d.down(y, H);
            q.memcpy(xr, x + size_t(r) * H, H * 2).wait();
            residual_add_rms_norm_fused(q, xr, x2 + size_t(r) * H, nw, y, 1, H, kEps).wait();
            sr[r] = d.down(y, H); sx[r] = d.down(xr, H);
        }
        for (uint32_t M : rows) {
            rms_norm_f32w(q, x, nw, y, M, H, kEps).wait();
            const auto b = d.down(y, size_t(M) * H);
            q.memcpy(xr, x, size_t(M) * H * 2).wait();
            residual_add_rms_norm_fused(q, xr, x2, nw, y, M, H, kEps).wait();
            const auto br = d.down(y, size_t(M) * H), bx = d.down(xr, size_t(M) * H);
            RowDiff d0, d1, d2;
            for (uint32_t r = 0; r < M; ++r) { row_diff(s[r], b, r, H, d0, false); row_diff(sr[r], br, r, H, d1, false); row_diff(sx[r], bx, r, H, d2, false); }
            check(!d0.rows_differ, "rms_norm_f32w M = " + std::to_string(M), fmt(d0, M, false));
            check(!d1.rows_differ && !d2.rows_differ, "residual_add_rms_norm_fused M = " + std::to_string(M) + " (normed and residual)",
                  fmt(d1, M, false) + " / " + fmt(d2, M, false));
        }
    }
    // quantize_q8_1: M rows at once == each row alone (the block stream's bytes)
    {
        const size_t rb = size_t(H / 32) * sizeof(block_q8_1x);
        std::vector<std::vector<uint8_t>> s(kMaxRows);
        for (uint32_t r = 0; r < kMaxRows; ++r) {
            quantize_q8_1(q, x + size_t(r) * H, act, H).wait();
            s[r] = d.down(reinterpret_cast<uint8_t*>(act), rb);
        }
        for (uint32_t M : rows) {
            quantize_q8_1(q, x, act, M * H).wait();
            const auto b = d.down(reinterpret_cast<uint8_t*>(act), rb * M);
            RowDiff dd; for (uint32_t r = 0; r < M; ++r) row_diff(s[r], b, r, rb, dd, false);
            check(!dd.rows_differ, "quantize_q8_1 M = " + std::to_string(M) + " rows (block_q8_1x bytes)", fmt(dd, M, false));
        }
    }
    // the dense int-dot GEMVs: batched (forward_slots' forced route) vs the crown's T == 1 route
    for (const Dense& sh : kDense) {
        Dev scratch_owner{d.q, {}};
        const Soa w = scratch_owner.soa(sh.K, sh.N, 100 + sh.N + sh.K);
        std::vector<std::vector<sycl::half>> s(kMaxRows);
        for (uint32_t r = 0; r < kMaxRows; ++r) { dense_rows(q, x + size_t(r) * sh.K, w, y, 1, act).wait(); s[r] = d.down(y, sh.N); }
        for (uint32_t M : rows) {
            dense_rows(q, x, w, y, M, act).wait();
            const auto b = d.down(y, size_t(M) * sh.N);
            RowDiff dd; for (uint32_t r = 0; r < M; ++r) row_diff(s[r], b, r, sh.N, dd, false);
            check(!dd.rows_differ, "int-dot " + std::string(sh.name) + " K " + std::to_string(sh.K) + " N " + std::to_string(sh.N) +
                  ": batched M = " + std::to_string(M) + " == the T == 1 kernel", fmt(dd, M, false));
        }
    }
    // alpha / beta (F16, N padded to 64) through dense::gemv_q_T: REPORTED (the GEMM route at T >= 2 may round differently)
    {
        auto* wab = d.alloc<sycl::half>(size_t(H) * AB_N);
        fill_dev(q, wab, size_t(H) * AB_N, 7, 0.05f);
        const DenseQuantPtr W{wab, DType::kF16};
        std::vector<std::vector<sycl::half>> s(kMaxRows);
        for (uint32_t r = 0; r < kMaxRows; ++r) { dense::gemv_q_T(q, x + size_t(r) * H, W, y, H, AB_N, 1).wait(); s[r] = d.down(y, AB_N); }
        for (uint32_t M : rows) {
            dense::gemv_q_T(q, x, W, y, H, AB_N, M).wait();
            const auto b = d.down(y, size_t(M) * AB_N);
            RowDiff dd; for (uint32_t r = 0; r < M; ++r) row_diff(s[r], b, r, AB_N, dd, false);
            info("alpha/beta gemv_q_T F16 2048 x 64 M = " + std::to_string(M) + " vs T == 1 (report only: a row step can run it per row)", fmt(dd, M, false));
        }
    }
    // the MoE Q8_0 decode chain, stage by stage
    {
        const Moe m = make_moe(d, 200);
        const MoeWs s = make_moe_ws(d);
        rms_norm_f32w(q, x, nw, xr, kMaxRows, H, kEps).wait();   // the router's input: normed rows
        struct Stage { std::vector<int32_t> idx; std::vector<sycl::half> w, h, yp, y; std::vector<uint8_t> xq, hq; };
        auto grab = [&](uint32_t G) {
            Stage st;
            st.idx = d.down(s.idx, size_t(G) * TK); st.w = d.down(s.w, size_t(G) * TK);
            st.xq = d.down(reinterpret_cast<uint8_t*>(s.xq8), size_t(G) * (H / 32) * sizeof(block_q8_1x));
            st.h = d.down(s.h, size_t(G) * TK * EFF);
            st.hq = d.down(reinterpret_cast<uint8_t*>(s.hq8), size_t(G) * TK * (EFF / 32) * sizeof(block_q8_1x));
            st.yp = d.down(s.yp, size_t(G) * TK * H); st.y = d.down(s.y, size_t(G) * H);
            return st;
        };
        std::vector<Stage> solo(kMaxRows);
        for (uint32_t r = 0; r < kMaxRows; ++r) { moe_chain(q, m, s, xr + size_t(r) * H, 1); q.wait(); solo[r] = grab(1); }
        for (uint32_t M : rows) {
            moe_chain(q, m, s, xr, M); q.wait();
            const Stage b = grab(M);
            RowDiff di, dw, dx, dh, dhq, dy, dr;
            for (uint32_t r = 0; r < M; ++r) {
                row_diff(solo[r].idx, b.idx, r, TK, di, false); row_diff(solo[r].w, b.w, r, TK, dw, false);
                row_diff(solo[r].xq, b.xq, r, solo[r].xq.size(), dx, false);
                row_diff(solo[r].h, b.h, r, size_t(TK) * EFF, dh, false); row_diff(solo[r].hq, b.hq, r, solo[r].hq.size(), dhq, false);
                row_diff(solo[r].yp, b.yp, r, size_t(TK) * H, dy, false); row_diff(solo[r].y, b.y, r, H, dr, false);
            }
            const std::string ms = " M = " + std::to_string(M);
            check(!di.rows_differ && !dw.rows_differ, "moe_router" + ms + " (top-8 ids and weights)", fmt(di, M, false) + " / " + fmt(dw, M, false));
            check(!dx.rows_differ, "quantize_q8_1 of the MoE input" + ms, fmt(dx, M, false));
            check(!dh.rows_differ, "moe_gate_up_silu_q8" + ms, fmt(dh, M, false));
            check(!dhq.rows_differ, "quantize_q8_1 of h (" + std::to_string(TK) + " slots a row)" + ms, fmt(dhq, M, false));
            check(!dy.rows_differ, "moe_down_q8" + ms, fmt(dy, M, false));
            check(!dr.rows_differ, "moe_reduce_q8" + ms, fmt(dr, M, false));
            // (report only) the fused T > 1 router against the T == 1 route: the kernel the rows path must NOT use
            moe_router(q, xr, m.router, s.idx, s.w, M, H, E, TK); q.wait();
            const std::vector<int32_t> fi = d.down(s.idx, size_t(M) * TK);
            const std::vector<sycl::half> fw = d.down(s.w, size_t(M) * TK);
            RowDiff fdi, fdw;
            for (uint32_t r = 0; r < M; ++r) { row_diff(solo[r].idx, fi, r, TK, fdi, false); row_diff(solo[r].w, fw, r, TK, fdw, false); }
            std::printf("[info] fused moe_router%s vs the T == 1 route (report only): %s / %s\n", ms.c_str(), fmt(fdi, M, false).c_str(),
                        fmt(fdw, M, false).c_str());
        }
    }
    // the head: rms_norm + the vocab GEMV
    {
        Dev scratch_owner{d.q, {}};
        const Soa w = scratch_owner.soa(H, V, 300);
        std::vector<std::vector<sycl::half>> s(kMaxRows);
        for (uint32_t r = 0; r < kMaxRows; ++r) {
            rms_norm_f32w(q, x + size_t(r) * H, nw, xr, 1, H, kEps);
            dense_rows(q, xr, w, y, 1, act).wait();
            s[r] = d.down(y, V);
        }
        for (uint32_t M : rows) {
            rms_norm_f32w(q, x, nw, xr, M, H, kEps);
            dense_rows(q, xr, w, y, M, act).wait();
            const auto b = d.down(y, size_t(M) * V);
            RowDiff dd; for (uint32_t r = 0; r < M; ++r) row_diff(s[r], b, r, V, dd, true);
            check(!dd.rows_differ, "head (rms_norm + int-dot 2048 x 248,320) M = " + std::to_string(M), fmt(dd, M, true));
        }
    }
}

// ---- --timing ------------------------------------------------------------------------------------------------------------
void timing(Dev& d, uint32_t layers, uint32_t reps, bool head) {
    sycl::queue& q = d.q;
    struct Layer { bool full; Soa a, b, c, e; sycl::half* ab; Soa shg, shu, shd; Moe moe; float* n1; float* n2; };
    std::vector<Layer> L(layers);
    const auto t_alloc = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < layers; ++i) {
        Layer& l = L[i];
        l.full = i % 4 == 3;
        const uint32_t sd = 1000 + 100 * i;
        if (l.full) { l.a = d.soa(H, N_QG, sd); l.b = d.soa(H, N_KV, sd + 1); l.c = d.soa(H, N_KV, sd + 2); l.e = d.soa(N_Q, H, sd + 3); }
        else { l.a = d.soa(H, CONV_CH, sd); l.b = d.soa(H, SI, sd + 1); l.c = d.soa(SI, H, sd + 2); }
        l.ab = d.alloc<sycl::half>(size_t(H) * AB_N * 2);
        fill_dev(q, l.ab, size_t(H) * AB_N * 2, sd + 4, 0.05f);
        l.shg = d.soa(H, SH_FF, sd + 5); l.shu = d.soa(H, SH_FF, sd + 6); l.shd = d.soa(SH_FF, H, sd + 7);
        l.moe = make_moe(d, sd + 8);
        l.n1 = d.alloc<float>(H); l.n2 = d.alloc<float>(H);
        fill_dev(q, l.n1, H, sd + 9, 0.2f, 1.f); fill_dev(q, l.n2, H, sd + 10, 0.2f, 1.f);
    }
    Soa hw; float* hn = nullptr;
    if (head) { hw = d.soa(H, V, 77); hn = d.alloc<float>(H); fill_dev(q, hn, H, 78, 0.2f, 1.f); }
    info("timing weights", std::to_string(layers) + " layers" + (head ? " + head" : "") + " uploaded in " +
         std::to_string(int(std::chrono::duration<double>(std::chrono::steady_clock::now() - t_alloc).count())) + " s");
    auto* x0 = d.alloc<sycl::half>(size_t(kMaxRows) * H); fill_dev(q, x0, size_t(kMaxRows) * H, 5, 1.f);
    auto* x = d.alloc<sycl::half>(size_t(kMaxRows) * H);
    auto* xn = d.alloc<sycl::half>(size_t(kMaxRows) * H);
    auto* big = d.alloc<sycl::half>(size_t(kMaxRows) * N_QG);
    auto* mid = d.alloc<sycl::half>(size_t(kMaxRows) * SI);
    auto* kv = d.alloc<sycl::half>(size_t(kMaxRows) * N_KV);
    auto* blk = d.alloc<sycl::half>(size_t(kMaxRows) * H);
    auto* ab = d.alloc<sycl::half>(size_t(kMaxRows) * AB_N);
    auto* g5 = d.alloc<sycl::half>(size_t(kMaxRows) * SH_FF); auto* u5 = d.alloc<sycl::half>(size_t(kMaxRows) * SH_FF);
    auto* h5 = d.alloc<sycl::half>(size_t(kMaxRows) * SH_FF); auto* eo = d.alloc<sycl::half>(size_t(kMaxRows) * H);
    auto* logits = d.alloc<sycl::half>(size_t(kMaxRows) * V);
    auto* act = d.alloc<block_q8_1x>(size_t(kMaxRows) * SI / 32);
    fill_dev(q, mid, size_t(kMaxRows) * SI, 6, 1.f);
    const MoeWs ms = make_moe_ws(d);

    auto step = [&](uint32_t G) {
        for (const Layer& l : L) {
            q.memcpy(x, x0, size_t(G) * H * 2);
            rms_norm_f32w(q, x, l.n1, xn, G, H, kEps);
            if (l.full) {
                dense_rows(q, xn, l.a, big, G, act); dense_rows(q, xn, l.b, kv, G, act); dense_rows(q, xn, l.c, kv, G, act);
                dense_rows(q, mid, l.e, blk, G, act);
            } else {
                dense_rows(q, xn, l.a, big, G, act);
                for (uint32_t r = 0; r < G; ++r)   // alpha / beta per row (the exact route; N = 64)
                    for (uint32_t k = 0; k < 2; ++k)
                        dense::gemv_q_T(q, xn + size_t(r) * H, DenseQuantPtr{l.ab + size_t(k) * H * AB_N, DType::kF16}, ab + size_t(r) * AB_N, H, AB_N, 1);
                dense_rows(q, xn, l.b, mid, G, act); dense_rows(q, mid, l.c, blk, G, act);
            }
            residual_add_rms_norm_fused(q, x, blk, l.n2, xn, G, H, kEps);
            moe_chain(q, l.moe, ms, xn, G);
            dense_rows(q, xn, l.shg, g5, G, act); dense_rows(q, xn, l.shu, u5, G, act);
            swiglu(q, g5, u5, h5, uint64_t(G) * SH_FF);
            dense_rows(q, h5, l.shd, eo, G, act);
        }
        if (head) { rms_norm_f32w(q, x, hn, xn, G, H, kEps); dense_rows(q, xn, hw, logits, G, act); }
        q.wait();
    };
    const uint32_t nfull = uint32_t(std::count_if(L.begin(), L.end(), [](const Layer& l) { return l.full; }));
    const double dense_b = double(layers - nfull) * dense_layer_bytes(false) + double(nfull) * dense_layer_bytes(true) + (head ? double(soa_bytes(H, V)) : 0.0);
    const double exp_b = double(2 * soa_bytes(H, EFF) + soa_bytes(EFF, H));
    std::printf("\n  G    ms/step   t(G)/t(1)  ms/row(marg)  distinct(L0)  GB/s(eff)   [p10 .. p90 ms]\n");
    std::vector<double> med(kMaxRows + 1, 0.0);
    for (uint32_t G = 1; G <= kMaxRows; ++G) {
        for (int w = 0; w < 3; ++w) step(G);
        std::vector<double> t;
        for (uint32_t r = 0; r < reps; ++r) {
            const auto a = std::chrono::steady_clock::now();
            step(G);
            t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
        }
        med[G] = pctl(t, 50);
        // the distinct experts layer 0's G rows picked (the last step's routing)
        (G == 1 ? moe_router(q, xn, L[0].moe.router, ms.idx, ms.w, 1, H, E, TK)
                : moe_router_rows(q, xn, L[0].moe.router, ms.idx, ms.w, G, H, E, TK)).wait();   // (xn = the head's input after the step; ids only)
        const auto ids = d.down(ms.idx, size_t(G) * TK);
        const std::set<int32_t> uniq(ids.begin(), ids.end());
        const double bytes = dense_b + double(layers) * expected_distinct(G) * exp_b;
        std::printf("  %2u  %9.3f   %8.3f   %10.3f   %6zu (%5.1f)   %8.1f   [%.3f .. %.3f]\n", G, med[G], med[G] / med[1],
                    G > 1 ? med[G] - med[G - 1] : 0.0, uniq.size(), expected_distinct(G), bytes / (med[G] * 1e6), pctl(t, 10), pctl(t, 90));
        std::fflush(stdout);
    }
    const double k4 = med[4] / med[1];
    std::printf("\nphase 1a kill criterion t(4)/t(1) > 3: %.3f -> %s\n", k4, k4 > 3.0 ? "KILL (row batching does not pay on this card)" : "pass");
    std::printf("(shared-weight part only: the per-lane conv / DeltaNet / attention are NOT in these steps)\n");
}

}  // namespace

int main(int argc, char** argv) {
    bool do_dry = false, do_self = false, do_exact = false, do_timing = false, head = true;
    std::vector<uint32_t> rows;
    for (uint32_t m = 2; m <= kMaxRows; ++m) rows.push_back(m);
    uint32_t layers = 20, reps = 10, device = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--dry") do_dry = true;
        else if (a == "--selftest") do_self = true;
        else if (a == "--exact") do_exact = true;
        else if (a == "--timing") do_timing = true;
        else if (a == "--no-head") head = false;
        else if (a == "--rows") rows = parse_list(val());
        else if (a == "--layers") layers = uint32_t(std::atoi(val().c_str()));
        else if (a == "--reps") reps = uint32_t(std::atoi(val().c_str()));
        else if (a == "--device") device = uint32_t(std::atoi(val().c_str()));
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    for (uint32_t m : rows) if (m < 1 || m > kMaxRows) { std::fprintf(stderr, "--rows: each must be 1..%u\n", kMaxRows); return 2; }
    if (layers < 1 || reps < 1) { std::fprintf(stderr, "--layers and --reps must be >= 1\n"); return 2; }
    if (do_self) { const int f = selftest(); std::printf("\nQ35M ROWS SELFTEST: %s\n", f ? "FAIL" : "PASS"); if (!do_dry && !do_exact && !do_timing) return f ? 1 : 0; }
    if (do_dry) { dry(rows, layers, head); return g_fail ? 1 : 0; }
    if (!do_exact && !do_timing) do_exact = do_timing = true;

    std::vector<sycl::device> gpus;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& dv : p.get_devices(sycl::info::device_type::gpu)) gpus.push_back(dv);
    }
    if (device >= gpus.size()) { std::printf("SKIP: no Level Zero GPU %u (%zu found)\n", device, gpus.size()); return 77; }
    Dev d{sycl::queue(sycl::context(gpus[device]), gpus[device], sycl::property_list{sycl::property::queue::in_order{}}), {}};
    std::printf("device %u: %s\n", device, gpus[device].get_info<sycl::info::device::name>().c_str());
    if (do_exact) exact(d, rows);
    if (do_timing) timing(d, layers, reps, head);
    std::printf("\nQ35M ROWS TEST: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
