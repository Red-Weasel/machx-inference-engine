// tools/q4e_rows_test.cpp -- ie-q4e-rows-test: P4 B17 (~/ds41_work/p60/b17). Can Flash-Next (qwen4exp, the two-card split)
// run G request lanes' decode rows as ONE pass over a stage (Qwen4ExpModel::forward_rows), every row bit-identical to that
// lane's one-row step (forward_range, T = 1), and what does a group step cost against G one-row steps?
//
// The model loads like `ie serve` loads it: stage A = blocks [0, n/2) on --devices' first card, stage B = [n/2, n) on the
// second, --lanes request lanes per stage (lane 0 and lanes 1.. all at --ctx), prefill chunks of 2048.
//
// --ops (real weights of stage A's blocks 0 / 1 (PLE) / 3 and stage B's blocks n/2 / n/2+3 and head; random activations):
//   every op a row group shares between lanes, at rows M = --rows (default every M in 2..16), compared ROW BY ROW, byte for
//   byte, with the route the one-row step runs for that row today:
//     * the int-dot GEMVs: quantize_q8_1 over M rows + gemv_q8_0_soa_q8_batched vs quantize_q8_1 + gemv_q8_0_soa_q8_g, at every
//       Q8-SoA tensor of the blocks (DeltaNet qkv / gate / out, attn q / output, the head) and, for the shared expert (whose
//       one-row route is gemv_q8_0_soa_q8_batched at T = 1), batched M vs batched 1;
//     * the F16 GEMVs: gemv_fp16_rows in <= 8-row chunks vs gemv_fp16 (attn k / v, the indexer q / k, PLE key / value, any
//       F16 fallback) and vs gemv_fp16_rows(R = 1) for the router (its one-row route); gemv_fp16_rows_dual (alpha / beta) and
//       gemv_fp16_dotrows (the shared-expert gate) M vs 1;
//     * the HC mixes (attn / ffn site with inject, the final merge) in <= 4-row chunks vs T = 1 -- plus, REPORTED only, the
//       unchunked M-row call (qwen4_hc_mix_v2 switches from the v3 grid to the v2 grid above 4 rows: the trap the chunks
//       avoid); qwen4_hc_combine, qwen4_hc_expand_streams;
//     * quantize_q8_1 / quantize_q8_1s over M rows;
//     * the MoE per-pick kernels on real expert slices (a block's raw Q4_K gate / up and Q5_1 down copied into 4 slots):
//       gemv_q4_K_q8s_grouped_dual and gemv_q5_1_grouped over M rows' picks vs one row's picks;
//     * the row-wise ops: rms_norm_f32w (q / k / indexer-q widths), rope_partial with a DIFFERENT position per row,
//       split_q_gate_per_head, sigmoid_gate, cast_qkv_split_fp16_to_fp32, l2_norm_scale, repeat_interleave_heads,
//       compute_g_beta_h16, gated_rms_norm, swiglu.
//   A difference in a must-match op is a FAIL (exit 1).
// --model (the end-to-end identity, real prompts, real routing -- the B14 lesson: random data hid the crown's router bug):
//   TWIN lanes: lanes [0, K) and [K, 2K) are prefilled with the same K prompts (short ones, one of --long tokens that
//   crosses the dense -> QSA switch at position 2051 while decoding, one past it). Then, for every M in --rows, --steps
//   steps: lanes [0, M) (rotated order, the live lane in the group on odd steps) run as ONE forward_rows group on both
//   stages; their twins run M one-row forward_range steps. Stage A's wide rows and stage B's logits must be byte-identical;
//   the next id is the one-row logits' argmax. At the end every lane's whole state (stash_slot on both stages: KV, indexer
//   caches, DeltaNet, PLE, histories) must equal its twin's. Timing: the group step (A + B, serial) against the M one-row
//   steps, and the expert-cache misses of each.
// --selftest: host checks (no device). Default: --ops then --model.
//   usage: ie-q4e-rows-test <model.gguf> [--selftest] [--ops] [--model] [--exact] [--rows 2,3,...,16] [--steps 6] [--ctx 4096]
//                           [--long 2046] [--devices 0,1]
#include "ie/allocator.hpp"
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"
#include "ie/ops.hpp"
#include "ie/q4e_lanes.hpp"
#include "ie/quant_blocks.hpp"
#include "ie/qwen4_hc.hpp"
#include "ie/qwen4_quant.hpp"
#include "ie/qwen4exp.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace ie;

namespace {

int g_fail = 0, g_ok = 0;
void check(bool ok, const std::string& what, const std::string& detail = "") {
    std::printf("[%s] %s%s%s\n", ok ? " ok " : "FAIL", what.c_str(), detail.empty() ? "" : "  ", detail.c_str());
    std::fflush(stdout);
    if (ok) ++g_ok; else ++g_fail;
}
void info(const std::string& what, const std::string& detail) { std::printf("[info] %s  %s\n", what.c_str(), detail.c_str()); std::fflush(stdout); }

std::vector<uint32_t> parse_rows(const std::string& s) {   // "2,3,5" or "2-16"
    std::vector<uint32_t> v;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        const std::string t = s.substr(i, j - i);
        const size_t d = t.find('-');
        if (d != std::string::npos) {
            const int a = std::atoi(t.substr(0, d).c_str()), b = std::atoi(t.substr(d + 1).c_str());
            for (int x = a; x <= b; ++x) v.push_back(uint32_t(x));
        } else if (!t.empty()) {
            v.push_back(uint32_t(std::atoi(t.c_str())));
        }
        i = j + 1;
    }
    return v;
}

enum class Ty { F16, F32, Raw };
// max |a - b| over one row of n elements of type ty (bytes already known to differ)
double row_max_abs(const uint8_t* a, const uint8_t* b, size_t row_bytes, Ty ty) {
    double m = 0;
    if (ty == Ty::F16) {
        const auto* x = reinterpret_cast<const sycl::half*>(a); const auto* y = reinterpret_cast<const sycl::half*>(b);
        for (size_t i = 0; i < row_bytes / 2; ++i) m = std::max(m, std::fabs(double(float(x[i])) - double(float(y[i]))));
    } else if (ty == Ty::F32) {
        const auto* x = reinterpret_cast<const float*>(a); const auto* y = reinterpret_cast<const float*>(b);
        for (size_t i = 0; i < row_bytes / 4; ++i) m = std::max(m, std::fabs(double(x[i]) - double(y[i])));
    } else {
        for (size_t i = 0; i < row_bytes; ++i) if (a[i] != b[i]) m = std::max(m, 1.0);
    }
    return m;
}

// Rows of `batched` against rows of `solo` (both M x row_bytes): rows that differ, max abs.
struct RowCmp { uint32_t differ = 0; double max_abs = 0; };
RowCmp cmp_rows(const std::vector<uint8_t>& batched, const std::vector<uint8_t>& solo, uint32_t M, size_t row_bytes, Ty ty) {
    RowCmp c;
    for (uint32_t r = 0; r < M; ++r) {
        const uint8_t* a = batched.data() + size_t(r) * row_bytes;
        const uint8_t* b = solo.data() + size_t(r) * row_bytes;
        if (std::memcmp(a, b, row_bytes) == 0) continue;
        ++c.differ;
        c.max_abs = std::max(c.max_abs, row_max_abs(a, b, row_bytes, ty));
    }
    return c;
}
std::string fmt(const RowCmp& c, uint32_t M) {
    if (!c.differ) return "bit-identical";
    char b[160];
    std::snprintf(b, sizeof b, "%u of %u rows differ, max abs %.3g", c.differ, M, c.max_abs);
    return b;
}

int32_t argmax_h(const sycl::half* p, uint32_t n) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < n; ++i) if (float(p[i]) > float(p[best])) best = i;
    return int32_t(best);
}

int selftest() {
    check(parse_rows("2-4,7") == std::vector<uint32_t>{2, 3, 4, 7}, "selftest: parse_rows ranges and lists");
    check(parse_rows("16") == std::vector<uint32_t>{16}, "selftest: parse_rows one value");
    std::vector<uint8_t> a(8, 0), b(8, 0);
    const sycl::half h1(1.0f), h2(1.5f);
    std::memcpy(b.data() + 4, &h2, 2); std::memcpy(a.data() + 4, &h1, 2);
    const RowCmp c = cmp_rows(a, b, 2, 4, Ty::F16);
    check(c.differ == 1 && std::fabs(c.max_abs - 0.5) < 1e-6, "selftest: cmp_rows finds row 1 only, max abs 0.5");
    check(cmp_rows(a, a, 2, 4, Ty::F16).differ == 0, "selftest: equal rows are bit-identical");
    check(q4e_rows_max_group(298, 10) == 16 && q4e_rows_max_group(64, 10) == 6, "selftest: q4e_rows_max_group");
    return g_fail ? 1 : 0;
}

// ------------------------------------------------------------------------------------------------------------------------
// --ops: one op = a launcher that writes rows [r0, r0 + rows) of its output at out + r * row_bytes. The batched call runs
// rows [0, M) at once (solo = false); the reference runs each row alone with the one-row route (solo = true).
using Launch = std::function<void(uint32_t r0, uint32_t rows, bool solo, uint8_t* out)>;

struct OpsBench {
    sycl::queue& q;
    std::vector<uint32_t> Ms;
    uint8_t* ob = nullptr;   // device [16 x max_row]
    uint8_t* os = nullptr;
    size_t cap = 0;
    OpsBench(sycl::queue& q_, std::vector<uint32_t> Ms_, size_t max_row) : q(q_), Ms(std::move(Ms_)), cap(16 * max_row) {
        ob = sycl::malloc_device<uint8_t>(cap, q);
        os = sycl::malloc_device<uint8_t>(cap, q);
    }
    ~OpsBench() { sycl::free(ob, q); sycl::free(os, q); }
    // report_only: print, never FAIL (the unchunked HC mix)
    void run(const std::string& name, size_t row_bytes, Ty ty, const Launch& fn, bool report_only = false) {
        if (16 * row_bytes > cap) { check(false, name, "row " + std::to_string(row_bytes) + " B exceeds the bench buffer"); return; }
        for (uint32_t M : Ms) {
            q.memset(ob, 0, M * row_bytes);
            q.memset(os, 0, M * row_bytes);
            fn(0, M, false, ob);
            for (uint32_t r = 0; r < M; ++r) fn(r, 1, true, os);
            q.wait();
            std::vector<uint8_t> hb(M * row_bytes), hs(M * row_bytes);
            q.memcpy(hb.data(), ob, hb.size());
            q.memcpy(hs.data(), os, hs.size()).wait();
            const RowCmp c = cmp_rows(hb, hs, M, row_bytes, ty);
            const std::string what = name + " M=" + std::to_string(M);
            if (report_only) info(what, fmt(c, M) + " (report only)");
            else check(c.differ == 0, what, fmt(c, M));
        }
    }
};

void fill_half(sycl::queue& q, sycl::half* d, size_t n, uint32_t seed, float scale) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, scale);
    std::vector<sycl::half> h(n);
    for (auto& x : h) x = sycl::half(nd(rng));
    q.memcpy(d, h.data(), n * 2).wait();
}
void fill_float(sycl::queue& q, float* d, size_t n, uint32_t seed, float scale) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, scale);
    std::vector<float> h(n);
    for (auto& x : h) x = nd(rng);
    q.memcpy(d, h.data(), n * 4).wait();
}

// Every shared op of one stage (its blocks L_dn (DeltaNet), L_fa (full attention), optionally L_ple, and the head when tail).
void ops_stage(const char* tag, Qwen4ExpModel& m, const Qwen4ExpConfig& c, uint32_t L_dn, uint32_t L_fa, int L_ple, bool tail,
               const std::vector<uint32_t>& Ms) {
    sycl::queue& q = m.queue();
    const uint32_t H = c.hidden, D = c.hc_count * H, R = c.hc_low_rank, HC = c.hc_count;
    const uint32_t SI = c.ssm_inner, SKH = c.ssm_k_heads, SVH = c.ssm_v_heads, SHD = c.ssm_state;
    const uint32_t CC = SI + 2u * SKH * SHD, KW = SKH * SHD;
    const uint32_t HD = c.head_dim, NQ = c.n_q_heads * HD, NKV = c.n_kv_heads * HD;
    const uint32_t EF = c.expert_ffn, SEF = c.shared_expert_ffn, NU = c.n_experts_used, V = c.vocab;
    const uint32_t IHD = c.indexer_head_dim, IQW = c.indexer_n_heads * IHD;
    const float eps = c.rms_eps;
    const std::string T = std::string(tag) + " ";
    const size_t KMAX = std::max<size_t>({H, SI, NQ * 2, CC, D});
    // inputs (16 rows each) and scratch
    auto* X16 = sycl::malloc_device<sycl::half>(16 * KMAX, q);
    auto* Y16 = sycl::malloc_device<sycl::half>(16 * KMAX, q);
    auto* W32 = sycl::malloc_device<float>(16 * KMAX, q);
    auto* Z32 = sycl::malloc_device<float>(16 * KMAX, q);
    auto* S16 = sycl::malloc_device<sycl::half>(16 * size_t(NU) * std::max(EF, H), q);   // scratch outputs
    auto* S32 = sycl::malloc_device<float>(16 * KMAX, q);
    auto* S32b = sycl::malloc_device<float>(16 * KMAX, q);
    auto* xn = sycl::malloc_device<float>(16 * size_t(D), q);
    auto* lo = sycl::malloc_device<float>(16 * size_t(R), q);
    auto* actB = sycl::malloc_device<block_q8_1x>(16 * (KMAX / 32), q);
    auto* actS = sycl::malloc_device<block_q8_1x>(KMAX / 32, q);
    auto* act8s = sycl::malloc_device<block_q8_1s>(16 * (H / 32), q);
    auto* posd = sycl::malloc_device<int32_t>(16, q);
    fill_half(q, X16, 16 * KMAX, 11, 1.0f);
    fill_half(q, Y16, 16 * KMAX, 12, 1.0f);
    fill_float(q, W32, 16 * KMAX, 13, 1.0f);
    fill_float(q, Z32, 16 * KMAX, 14, 0.5f);
    {
        int32_t p[16];
        for (int r = 0; r < 16; ++r) p[r] = 17 + 131 * r;   // a different position in every row
        q.memcpy(posd, p, sizeof p).wait();
    }
    OpsBench B(q, Ms, std::max<size_t>({size_t(V) * 2, size_t(D) * 4, size_t(NU) * std::max(EF, H) * 2, size_t(KMAX) * 4}));

    auto q8op = [&](const std::string& nm, const Qwen4ExpLayer::Q8W& W, uint32_t K, uint32_t N, bool solo_batched1) {
        if (!W.qs) { info(T + nm, "not Q8-SoA in this file (its F16 route is checked below if it exists)"); return; }
        B.run(T + nm + (solo_batched1 ? " int-dot batched M vs batched 1" : " int-dot batched vs _g"), size_t(N) * 2, Ty::F16,
              [&](uint32_t r0, uint32_t rows, bool solo, uint8_t* out) {
                  sycl::half* y = reinterpret_cast<sycl::half*>(out) + size_t(r0) * N;
                  const sycl::half* x = X16 + size_t(r0) * K;
                  if (solo && !solo_batched1) {
                      quantize_q8_1(q, x, actS, K);
                      gemv_q8_0_soa_q8_g(q, actS, W.qs, W.d, y, K, N);
                  } else {
                      void* act = solo ? static_cast<void*>(actS) : static_cast<void*>(actB);
                      quantize_q8_1(q, x, act, rows * K);
                      gemv_q8_0_soa_q8_batched(q, act, W.qs, W.d, y, K, N, rows);
                  }
              });
    };
    auto f16op = [&](const std::string& nm, const sycl::half* W, uint32_t K, uint32_t N, bool solo_rows1) {
        if (!W) return;
        B.run(T + nm + (solo_rows1 ? " fp16 rows(<=8) vs rows(1)" : " fp16 rows(<=8) vs gemv_fp16"), size_t(N) * 2, Ty::F16,
              [&](uint32_t r0, uint32_t rows, bool solo, uint8_t* out) {
                  sycl::half* y = reinterpret_cast<sycl::half*>(out) + size_t(r0) * N;
                  const sycl::half* x = X16 + size_t(r0) * K;
                  if (solo) {
                      if (solo_rows1) gemv_fp16_rows(q, x, K, W, y, N, K, N, 1);
                      else            gemv_fp16(q, x, W, y, K, N);
                  } else {
                      for (uint32_t c0 = 0; c0 < rows; c0 += 8)
                          gemv_fp16_rows(q, x + size_t(c0) * K, K, W, y + size_t(c0) * N, N, K, N, std::min(8u, rows - c0));
                  }
              });
    };
    // HC mix: `mixed` rows (out = mixed) or inject-logit rows (out = inj); chunks <= 4 (the rows path) or unchunked
    auto hcop = [&](const std::string& nm, const float* norm, const sycl::half* down, const sycl::half* up, const float* inj,
                    bool want_inj, bool chunked) {
        if (want_inj && !inj) return;
        const size_t rb = want_inj ? size_t(HC) * 4 : size_t(H) * 2;
        B.run(T + nm + (want_inj ? " (inject logits)" : " (mixed)") + (chunked ? " chunks<=4 vs T=1" : " UNCHUNKED vs T=1"), rb,
              want_inj ? Ty::F32 : Ty::F16,
              [&](uint32_t r0, uint32_t rows, bool solo, uint8_t* out) {
                  const uint32_t step = (solo || chunked) ? 4u : rows;
                  for (uint32_t c0 = 0; c0 < rows; c0 += step) {
                      const uint32_t r = r0 + c0, n = std::min(step, rows - c0);
                      sycl::half* mixed = want_inj ? S16 + size_t(r) * H : reinterpret_cast<sycl::half*>(out) + size_t(r) * H;
                      float* il = inj ? (want_inj ? reinterpret_cast<float*>(out) + size_t(r) * HC : S32 + size_t(r) * HC) : nullptr;
                      qwen4_hc_mix_v2(q, W32 + size_t(r) * D, norm, down, up, inj, xn + size_t(r) * D, lo + size_t(r) * R, mixed,
                                      il, n, H, HC, R, eps);
                  }
              }, /*report_only=*/!chunked);
    };

    const Qwen4ExpLayer& wd = m.layer(L_dn);
    const Qwen4ExpLayer& wf = m.layer(L_fa);
    std::printf("\n== %s: DeltaNet block %u, full-attention block %u%s%s\n", tag, L_dn, L_fa,
                L_ple >= 0 ? (", PLE block " + std::to_string(L_ple)).c_str() : "", tail ? ", head" : "");

    // -- dense projections --
    q8op("blk" + std::to_string(L_dn) + " attn_qkv", wd.qkv_q8, H, CC, false);
    f16op("blk" + std::to_string(L_dn) + " attn_qkv (F16)", wd.qkv_q8.qs ? nullptr : wd.attn_qkv, H, CC, false);
    q8op("blk" + std::to_string(L_dn) + " attn_gate", wd.gate_q8, H, SI, false);
    f16op("blk" + std::to_string(L_dn) + " attn_gate (F16)", wd.gate_q8.qs ? nullptr : wd.attn_gate, H, SI, false);
    q8op("blk" + std::to_string(L_dn) + " ssm_out", wd.out_q8, SI, H, false);
    f16op("blk" + std::to_string(L_dn) + " ssm_out (F16)", wd.out_q8.qs ? nullptr : wd.ssm_out, SI, H, false);
    q8op("blk" + std::to_string(L_fa) + " attn_q", wf.q_q8, H, NQ * 2, false);
    f16op("blk" + std::to_string(L_fa) + " attn_q (F16)", wf.q_q8.qs ? nullptr : wf.attn_q, H, NQ * 2, false);
    f16op("blk" + std::to_string(L_fa) + " attn_k", wf.attn_k, H, NKV, false);
    f16op("blk" + std::to_string(L_fa) + " attn_v", wf.attn_v, H, NKV, false);
    q8op("blk" + std::to_string(L_fa) + " attn_output", wf.attnout_q8, NQ, H, false);
    f16op("blk" + std::to_string(L_fa) + " attn_output (F16)", wf.attnout_q8.qs ? nullptr : wf.attn_output, NQ, H, false);
    f16op("blk" + std::to_string(L_fa) + " indexer q", wf.idx_q_proj, H, IQW, false);
    f16op("blk" + std::to_string(L_fa) + " indexer k", wf.idx_k_proj, H, IHD, false);
    if (L_ple >= 0) {
        const Qwen4ExpLayer& wp = m.layer(uint32_t(L_ple));
        f16op("blk" + std::to_string(L_ple) + " ple_key", wp.ple_key, H, D, false);
        f16op("blk" + std::to_string(L_ple) + " ple_value", wp.ple_value, H, H, false);
    }
    // -- MoE shared parts (block L_dn): router (one-row route = rows(R = 1)), shared expert (batched at T = 1), scalar gate --
    f16op("blk" + std::to_string(L_dn) + " router", wd.ffn_gate_inp, H, c.n_experts, true);
    q8op("blk" + std::to_string(L_dn) + " shexp gate", wd.shg_q8, H, SEF, true);
    q8op("blk" + std::to_string(L_dn) + " shexp up", wd.shu_q8, H, SEF, true);
    q8op("blk" + std::to_string(L_dn) + " shexp down", wd.shd_q8, SEF, H, true);
    f16op("blk" + std::to_string(L_dn) + " shexp gate (F16)", wd.shg_q8.qs ? nullptr : wd.ffn_gate_shexp, H, SEF, false);
    f16op("blk" + std::to_string(L_dn) + " shexp up (F16)", wd.shu_q8.qs ? nullptr : wd.ffn_up_shexp, H, SEF, false);
    f16op("blk" + std::to_string(L_dn) + " shexp down (F16)", wd.shd_q8.qs ? nullptr : wd.ffn_down_shexp, SEF, H, false);
    B.run(T + "blk" + std::to_string(L_dn) + " shexp scalar gate gemv_fp16_dotrows M vs 1", 2, Ty::F16,
          [&](uint32_t r0, uint32_t rows, bool, uint8_t* out) {
              gemv_fp16_dotrows(q, X16 + size_t(r0) * H, H, wd.ffn_gate_inp_shexp, reinterpret_cast<sycl::half*>(out) + r0, H, rows);
          });
    B.run(T + "blk" + std::to_string(L_dn) + " alpha/beta gemv_fp16_rows_dual M vs 1", 128 * 2, Ty::F16,
          [&](uint32_t r0, uint32_t rows, bool, uint8_t* out) {
              sycl::half* y = reinterpret_cast<sycl::half*>(out) + size_t(r0) * 128;
              gemv_fp16_rows_dual(q, X16 + size_t(r0) * H, H, wd.ssm_alpha, y, wd.ssm_beta, y + 64, 128, H, 64, rows);
          });
    // -- HC --
    for (int site = 0; site < 2; ++site) {
        const float* nrm = site ? wd.hc_ffn_norm : wd.hc_attn_norm;
        const sycl::half* dn = site ? wd.hc_ffn_down : wd.hc_attn_down;
        const sycl::half* up = site ? wd.hc_ffn_up : wd.hc_attn_up;
        const float* ij = site ? wd.hc_ffn_inject : wd.hc_attn_inject;
        const std::string nm = "blk" + std::to_string(L_dn) + (site ? " hc ffn mix" : " hc attn mix");
        hcop(nm, nrm, dn, up, ij, false, true);
        hcop(nm, nrm, dn, up, ij, true, true);
        hcop(nm, nrm, dn, up, ij, false, false);
    }
    if (tail) {
        hcop("final merge", m.out_hc_norm, m.out_hc_down, m.out_hc_up, nullptr, false, true);
        hcop("final merge", m.out_hc_norm, m.out_hc_down, m.out_hc_up, nullptr, false, false);
        q8op("head (vocab)", m.lmh_q8, H, V, false);
        f16op("head (vocab, F16)", m.lmh_q8.qs ? nullptr : m.lm_head, H, V, false);
    }
    B.run(T + "hc combine M vs 1", size_t(D) * 4, Ty::F32, [&](uint32_t r0, uint32_t rows, bool, uint8_t* out) {
        float* o = reinterpret_cast<float*>(out) + size_t(r0) * D;
        q.memcpy(o, W32 + size_t(r0) * D, size_t(rows) * D * 4);
        qwen4_hc_combine(q, o, X16 + size_t(r0) * H, Z32 + size_t(r0) * HC, rows, H, HC);
    });
    B.run(T + "hc expand streams M vs 1", size_t(D) * 4, Ty::F32, [&](uint32_t r0, uint32_t rows, bool, uint8_t* out) {
        qwen4_hc_expand_streams(q, X16 + size_t(r0) * H, reinterpret_cast<float*>(out) + size_t(r0) * D, rows, H, HC);
    });
    // -- activation quantizers --
    for (uint32_t K : {H, SI})
        B.run(T + "quantize_q8_1 K=" + std::to_string(K) + " M vs 1", size_t(K / 32) * sizeof(block_q8_1x), Ty::Raw,
              [&, K](uint32_t r0, uint32_t rows, bool, uint8_t* out) {
                  quantize_q8_1(q, X16 + size_t(r0) * K, out + size_t(r0) * (K / 32) * sizeof(block_q8_1x), rows * K);
              });
    B.run(T + "quantize_q8_1s K=" + std::to_string(H) + " M vs 1", size_t(H / 32) * sizeof(block_q8_1s), Ty::Raw,
          [&](uint32_t r0, uint32_t rows, bool, uint8_t* out) {
              quantize_q8_1s(q, X16 + size_t(r0) * H, out + size_t(r0) * (H / 32) * sizeof(block_q8_1s), uint64_t(rows) * H);
          });
    // -- MoE per-pick kernels on real expert slices (the first block of the stage with raw gate/up AND raw down banks) --
    {
        int Lm = -1;
        for (uint32_t L = L_dn; L < L_dn + 24 && L < c.n_layers; ++L)
            if (m.layer(L).gate_bv.raw && m.layer(L).down_bv.raw) { Lm = int(L); break; }
        if (Lm < 0) {
            info(T + "moe per-pick", "no block with raw Q4_K gate/up and Q5_1 down near block " + std::to_string(L_dn));
        } else {
            const Qwen4ExpLayer& wm = m.layer(uint32_t(Lm));
            const uint64_t goff = 0, uoff = wm.gate_bv.slice_bytes, doff = uoff + wm.up_bv.slice_bytes;
            const uint64_t sb = doff + wm.down_bv.slice_bytes;
            const uint32_t experts[4] = {7, 100, 250, c.n_experts - 1};
            auto* slots = sycl::malloc_device<uint8_t>(4 * sb, q);
            for (uint32_t s = 0; s < 4; ++s) {
                const uint32_t e = experts[s];
                q.memcpy(slots + s * sb + goff, wm.gate_bv.raw + uint64_t(e) * wm.gate_bv.slice_bytes, wm.gate_bv.slice_bytes);
                q.memcpy(slots + s * sb + uoff, wm.up_bv.raw + uint64_t(e) * wm.up_bv.slice_bytes, wm.up_bv.slice_bytes);
                q.memcpy(slots + s * sb + doff, wm.down_bv.raw + uint64_t(e) * wm.down_bv.slice_bytes, wm.down_bv.slice_bytes);
            }
            std::vector<int32_t> jobs(16 * NU * 3);
            for (uint32_t vt = 0; vt < 16; ++vt)
                for (uint32_t k = 0; k < NU; ++k) {
                    const uint32_t pk = vt * NU + k;
                    jobs[3 * pk + 0] = int32_t(vt);
                    jobs[3 * pk + 1] = int32_t((vt + k) % 4);
                    jobs[3 * pk + 2] = int32_t(pk);
                }
            auto* jd = sycl::malloc_device<int32_t>(jobs.size(), q);
            q.memcpy(jd, jobs.data(), jobs.size() * 4).wait();
            quantize_q8_1s(q, X16, act8s, uint64_t(16) * H);
            auto* hrows = sycl::malloc_device<sycl::half>(16 * size_t(NU) * EF, q);
            fill_half(q, hrows, 16 * size_t(NU) * EF, 21, 0.5f);
            const std::string nm = "blk" + std::to_string(Lm) + " moe ";
            for (int want_up = 0; want_up < 2; ++want_up)
                B.run(T + nm + (want_up ? "up" : "gate") + " gemv_q4_K_q8s_grouped_dual M rows' picks vs 1 row's",
                      size_t(NU) * EF * 2, Ty::F16, [&, want_up](uint32_t r0, uint32_t rows, bool, uint8_t* out) {
                          sycl::half* o = reinterpret_cast<sycl::half*>(out) + size_t(r0) * NU * EF;
                          sycl::half* s = S16 + size_t(r0) * NU * EF;
                          gemv_q4_K_q8s_grouped_dual(q, act8s, slots, sb, goff, uoff, jd + size_t(r0) * NU * 3,
                                                     want_up ? s : o, want_up ? o : s, H, EF, rows * NU);
                      });
            B.run(T + nm + "down gemv_q5_1_grouped M rows' picks vs 1 row's", size_t(NU) * H * 2, Ty::F16,
                  [&](uint32_t r0, uint32_t rows, bool, uint8_t* out) {
                      gemv_q5_1_grouped(q, hrows + size_t(r0) * NU * EF, slots, sb, doff, jd + size_t(r0) * NU * 3,
                                        reinterpret_cast<sycl::half*>(out), H, EF, H, rows * NU);
                  });
            q.wait();
            sycl::free(slots, q); sycl::free(jd, q); sycl::free(hrows, q);
        }
    }
    // -- row-wise ops --
    struct NormCase { const char* nm; const float* w; uint32_t heads, width; };
    const NormCase norms[] = {{"q norm", wf.attn_q_norm, c.n_q_heads, HD}, {"k norm", wf.attn_k_norm, c.n_kv_heads, HD},
                              {"indexer q norm", wf.idx_q_norm, c.indexer_n_heads, IHD}};
    for (const NormCase& nc : norms) {
        if (!nc.w) continue;
        const uint32_t n = nc.heads * nc.width;
        B.run(T + "rms_norm_f32w " + nc.nm + " M vs 1", size_t(n) * 2, Ty::F16, [&, n](uint32_t r0, uint32_t rows, bool, uint8_t* out) {
            rms_norm_f32w(q, X16 + size_t(r0) * n, nc.w, reinterpret_cast<sycl::half*>(out) + size_t(r0) * n, rows * nc.heads,
                          nc.width, eps);
        });
    }
    struct RopeCase { const char* nm; uint32_t heads, hd; };
    const RopeCase ropes[] = {{"q", c.n_q_heads, HD}, {"k", c.n_kv_heads, HD}, {"indexer q", c.indexer_n_heads, IHD}};
    for (const RopeCase& rc : ropes) {
        const uint32_t n = rc.heads * rc.hd;
        B.run(T + "rope_partial " + rc.nm + " (a position per row) M vs 1", size_t(n) * 2, Ty::F16,
              [&, n](uint32_t r0, uint32_t rows, bool, uint8_t* out) {
                  rope_partial(q, X16 + size_t(r0) * n, posd + r0, reinterpret_cast<sycl::half*>(out) + size_t(r0) * n, rows,
                               rc.heads, rc.hd, c.rope_dim, c.rope_theta);
              });
    }
    for (int part = 0; part < 2; ++part)
        B.run(T + "split_q_gate_per_head (" + (part ? "gate" : "q") + ") M vs 1", size_t(NQ) * 2, Ty::F16,
              [&, part](uint32_t r0, uint32_t rows, bool, uint8_t* out) {
                  sycl::half* o = reinterpret_cast<sycl::half*>(out) + size_t(r0) * NQ;
                  sycl::half* s = S16 + size_t(r0) * NQ;
                  split_q_gate_per_head(q, X16 + size_t(r0) * NQ * 2, part ? s : o, part ? o : s, rows, c.n_q_heads, HD);
              });
    B.run(T + "sigmoid_gate M vs 1", size_t(NQ) * 2, Ty::F16, [&](uint32_t r0, uint32_t rows, bool, uint8_t* out) {
        sigmoid_gate(q, X16 + size_t(r0) * NQ, Y16 + size_t(r0) * NQ, reinterpret_cast<sycl::half*>(out) + size_t(r0) * NQ,
                     uint64_t(rows) * NQ);
    });
    for (int part = 0; part < 3; ++part) {
        const uint32_t n = part == 2 ? SI : KW;
        B.run(T + "cast_qkv_split_fp16_to_fp32 (" + (part == 0 ? "q" : part == 1 ? "k" : "v") + ") M vs 1", size_t(n) * 4,
              Ty::F32, [&, part, n](uint32_t r0, uint32_t rows, bool, uint8_t* out) {
                  float* o = reinterpret_cast<float*>(out) + size_t(r0) * n;
                  float* qd = part == 0 ? o : S32 + size_t(r0) * KW;
                  float* kd = part == 1 ? o : S32b + size_t(r0) * KW;
                  float* vd = part == 2 ? o : Z32 + size_t(8) * KMAX + size_t(r0) * SI;   // (Z32's upper half as scratch)
                  cast_qkv_split_fp16_to_fp32(q, X16 + size_t(r0) * CC, qd, kd, vd, rows, KW, SI);
              });
    }
    B.run(T + "l2_norm_scale M vs 1", size_t(KW) * 4, Ty::F32, [&](uint32_t r0, uint32_t rows, bool, uint8_t* out) {
        l2_norm_scale(q, W32 + size_t(r0) * KW, reinterpret_cast<float*>(out) + size_t(r0) * KW, rows * SKH, SHD,
                      1.0f / std::sqrt(float(SHD)), 1e-6f);
    });
    B.run(T + "repeat_interleave_heads M vs 1", size_t(SI) * 4, Ty::F32, [&](uint32_t r0, uint32_t rows, bool, uint8_t* out) {
        repeat_interleave_heads(q, W32 + size_t(r0) * KW, reinterpret_cast<float*>(out) + size_t(r0) * SI, rows, SKH, SHD, SVH / SKH);
    });
    for (int part = 0; part < 2; ++part)
        B.run(T + "compute_g_beta_h16 (" + (part ? "beta" : "g") + ") M vs 1", size_t(SVH) * 4, Ty::F32,
              [&, part](uint32_t r0, uint32_t rows, bool, uint8_t* out) {
                  float* o = reinterpret_cast<float*>(out) + size_t(r0) * SVH;
                  float* s = S32 + size_t(r0) * SVH;
                  compute_g_beta_h16(q, X16 + size_t(r0) * SVH, Y16 + size_t(r0) * SVH, wd.ssm_a, wd.ssm_dt, part ? s : o,
                                     part ? o : s, rows, SVH);
              });
    B.run(T + "gated_rms_norm (sigmoid gate) M vs 1", size_t(SI) * 2, Ty::F16, [&](uint32_t r0, uint32_t rows, bool, uint8_t* out) {
        gated_rms_norm(q, W32 + size_t(r0) * SI, X16 + size_t(r0) * SI, wd.ssm_norm, reinterpret_cast<sycl::half*>(out) + size_t(r0) * SI,
                       rows * SVH, SHD, eps, true);
    });
    B.run(T + "swiglu M vs 1", size_t(SEF) * 2, Ty::F16, [&](uint32_t r0, uint32_t rows, bool, uint8_t* out) {
        swiglu(q, X16 + size_t(r0) * SEF, Y16 + size_t(r0) * SEF, reinterpret_cast<sycl::half*>(out) + size_t(r0) * SEF,
               uint64_t(rows) * SEF);
    });
    q.wait();
    for (void* p : std::initializer_list<void*>{X16, Y16, W32, Z32, S16, S32, S32b, xn, lo, actB, actS, act8s, posd}) sycl::free(p, q);
}

// ------------------------------------------------------------------------------------------------------------------------
// --model: twin lanes, real prompts.
struct Stages { Qwen4ExpModel* A; Qwen4ExpModel* B; };

std::string select(Stages& s, uint32_t lane) {
    if (auto e = s.A->select_lane(lane); !e.empty()) return e;
    return s.B->select_lane(lane);
}

// One lane's one-row step through both stages (what the per-lane pipe runs): its wide row and logits.
std::string solo_step(Stages& s, uint32_t lane, int32_t id, uint32_t pos, std::vector<float>& wide, std::vector<sycl::half>& lg) {
    if (auto e = select(s, lane); !e.empty()) return e;
    if (auto e = s.A->forward_range(&id, 1, pos, nullptr, wide.data(), nullptr); !e.empty()) return "A: " + e;
    if (auto e = s.B->forward_range(&id, 1, pos, wide.data(), nullptr, nullptr); !e.empty()) return "B: " + e;
    s.B->queue().memcpy(lg.data(), s.B->logits(), lg.size() * 2).wait();
    return {};
}

bool same_state(const Qwen4ExpModel::SlotState& a, const Qwen4ExpModel::SlotState& b, std::string& what) {
    auto eq = [&](const std::vector<uint8_t>& x, const std::vector<uint8_t>& y, const char* nm) {
        if (x == y) return true;
        what = nm;
        return false;
    };
    if (a.depth != b.depth) { what = "depth"; return false; }
    if (a.blk_done != b.blk_done) { what = "blk_done"; return false; }
    if (a.hist.h1 != b.hist.h1 || a.hist.h2 != b.hist.h2) { what = "PLE history"; return false; }
    return eq(a.kv_k, b.kv_k, "KV K") && eq(a.kv_v, b.kv_v, "KV V") && eq(a.idxk, b.idxk, "indexer raw keys") &&
           eq(a.blkk, b.blkk, "indexer pooled keys") && eq(a.dns, b.dns, "DeltaNet state") && eq(a.dnc, b.dnc, "DeltaNet conv") &&
           eq(a.ple, b.ple, "PLE conv");
}

int model_test(Stages& s, const Qwen4ExpConfig& c, const Tokenizer& tok, const std::vector<uint32_t>& Ms, uint32_t K,
               uint32_t steps, uint32_t long_len) {
    const uint32_t V = c.vocab, D = c.hc_count * c.hidden;
    std::printf("\n== model: twin lanes [0, %u) grouped / [%u, %u) one row a step; rows %zu sizes x %u steps\n", K, K, 2 * K,
                Ms.size(), steps);
    // K prompts: two long (one crossing the dense -> QSA switch at 2051 while decoding, one past it), the rest short
    const char* texts[] = {
        "Explain why the sky is blue in one short paragraph.",
        "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
        "List five uses of a paperclip that are not holding paper.",
        "What is the capital of Australia, and why is it not Sydney?",
        "Translate 'the quick brown fox jumps over the lazy dog' into French and German.",
        "Summarize the plot of Hamlet in three sentences.",
        "Give me a haiku about autumn rain.",
        "How does a hash map handle collisions? Answer briefly.",
        "Name three prime numbers greater than 100 and show why each is prime.",
        "Describe the water cycle to a ten-year-old.",
        "What are the main differences between TCP and UDP?",
        "Write a limerick about a cat who learns to code.",
        "Why do leaves change color in the fall?",
        "Convert 72 degrees Fahrenheit to Celsius and show the formula.",
        "Suggest a name for a bakery that also sells books.",
        "Explain recursion using a real-world analogy.",
    };
    std::vector<std::vector<int32_t>> prompts(K);
    const std::string filler = "The history of computing is long and varied, from mechanical calculators to modern processors. ";
    for (uint32_t i = 0; i < K; ++i) {
        const std::string t = std::string("<|im_start|>user\n") + texts[i % 16] + "<|im_end|>\n<|im_start|>assistant\n";
        std::vector<int32_t> ids = tok.encode(t, true);
        const uint32_t want = i == 0 ? long_len : i == 1 ? long_len + 300 : 0;
        if (want > ids.size()) {   // a long prompt: filler text before the question, cut to `want` tokens
            std::vector<int32_t> f = tok.encode(filler, true);
            std::vector<int32_t> lp;
            while (lp.size() + ids.size() < want) lp.insert(lp.end(), f.begin(), f.end());
            lp.resize(want - ids.size());
            lp.insert(lp.end(), ids.begin(), ids.end());
            ids.swap(lp);
        }
        prompts[i] = ids;
    }
    std::vector<float> wide(size_t(2048) * D);
    std::vector<uint32_t> pos(2 * K, 0);
    std::vector<int32_t> next(2 * K, 0);
    std::vector<std::vector<sycl::half>> plg(K, std::vector<sycl::half>(V));   // lanes [0, K)'s prompt logits
    std::vector<sycl::half> lg2(V);
    // prefill every lane (both twins the same way: the serve plan's 2048-row pieces through A then B)
    for (uint32_t l = 0; l < 2 * K; ++l) {
        const std::vector<int32_t>& ids = prompts[l % K];
        if (auto e = select(s, l); !e.empty()) { check(false, "select lane " + std::to_string(l), e); return 1; }
        s.A->reset_state(); s.B->reset_state();
        for (uint32_t p = 0; p < ids.size(); p += 2048) {
            const uint32_t n = std::min<uint32_t>(2048, uint32_t(ids.size()) - p);
            std::string e = s.A->forward_range(ids.data() + p, n, p, nullptr, wide.data(), nullptr);
            if (e.empty()) e = s.B->forward_range(ids.data() + p, n, p, wide.data(), nullptr, nullptr);
            if (!e.empty()) { check(false, "prefill lane " + std::to_string(l), e); return 1; }
        }
        sycl::half* dst = l < K ? plg[l].data() : lg2.data();
        s.B->queue().memcpy(dst, s.B->logits(), size_t(V) * 2).wait();
        pos[l] = uint32_t(ids.size());
        next[l] = argmax_h(dst, V);
        // a twin's prompt logits must equal its lane's byte for byte (deterministic prefill), or the comparison below is void
        if (l >= K && std::memcmp(lg2.data(), plg[l - K].data(), size_t(V) * 2) != 0) {
            check(false, "prefill twins agree", "lane " + std::to_string(l) + " vs " + std::to_string(l - K));
            return 1;
        }
    }
    check(true, "prefill: the " + std::to_string(K) + " twin pairs' prompt logits are byte-identical");
    info("prefill", "lanes 0/1 at " + std::to_string(pos[0]) + " / " + std::to_string(pos[1]) + " tokens, the rest " +
                    std::to_string(pos[2 % K]) + "..; dense attention ends at position 2051");

    std::vector<float> wideR(size_t(16) * D), wideS(D);
    std::vector<sycl::half> lgR(size_t(16) * V), lgS(V);
    for (uint32_t M : Ms) {
        if (M < 2 || M > K) { check(false, "rows M=" + std::to_string(M), "outside 2.." + std::to_string(K)); continue; }
        uint32_t bad_wide = 0, bad_lg = 0, argflip = 0;
        double t_rows = 0, t_solo = 0;
        uint64_t mr = 0, ms = 0;
        std::string first;
        for (uint32_t st = 0; st < steps; ++st) {
            uint32_t lanes[16], P[16];
            int32_t ids[16];
            for (uint32_t r = 0; r < M; ++r) {   // rotated order: row r = lane (r + st) % M
                lanes[r] = (r + st) % M;
                P[r] = pos[lanes[r]];
                ids[r] = next[lanes[r]];
            }
            if (st & 1) (void)select(s, lanes[M / 2]);   // odd steps: a lane of the group is the live lane
            const uint64_t m0 = s.A->ecache_misses + s.B->ecache_misses;
            const auto t0 = std::chrono::steady_clock::now();
            std::string e = s.A->forward_rows(lanes, ids, P, M, nullptr, wideR.data());
            if (e.empty()) e = s.B->forward_rows(lanes, ids, P, M, wideR.data(), nullptr);
            if (!e.empty()) { check(false, "forward_rows M=" + std::to_string(M), e); return 1; }
            s.B->queue().memcpy(lgR.data(), s.B->rows_logits(), size_t(M) * V * 2).wait();
            const auto t1 = std::chrono::steady_clock::now();
            const uint64_t m1 = s.A->ecache_misses + s.B->ecache_misses;
            t_rows += std::chrono::duration<double, std::milli>(t1 - t0).count();
            mr += m1 - m0;
            for (uint32_t r = 0; r < M; ++r) {
                const uint32_t l = lanes[r], tw = l + K;
                const auto u0 = std::chrono::steady_clock::now();
                if (auto e2 = solo_step(s, tw, ids[r], pos[tw], wideS, lgS); !e2.empty()) {
                    check(false, "one-row step lane " + std::to_string(tw), e2); return 1;
                }
                t_solo += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - u0).count();
                const bool wok = std::memcmp(wideR.data() + size_t(r) * D, wideS.data(), size_t(D) * 4) == 0;
                const bool lok = std::memcmp(lgR.data() + size_t(r) * V, lgS.data(), size_t(V) * 2) == 0;
                const int32_t amR = argmax_h(lgR.data() + size_t(r) * V, V), amS = argmax_h(lgS.data(), V);
                if (!wok) ++bad_wide;
                if (!lok) ++bad_lg;
                if (amR != amS) ++argflip;
                if ((!wok || !lok) && first.empty())
                    first = "first at step " + std::to_string(st) + " row " + std::to_string(r) + " (lane " + std::to_string(l) +
                            " @" + std::to_string(pos[l]) + ")" + (wok ? "" : " stage-A wide") + (lok ? "" : " logits");
                next[l] = next[tw] = amS;
                ++pos[l]; ++pos[tw];
            }
            ms += (s.A->ecache_misses + s.B->ecache_misses) - m1;
        }
        const std::string what = "model M=" + std::to_string(M) + ": " + std::to_string(steps) + " group steps == one-row steps";
        char det[320];
        std::snprintf(det, sizeof det, "wide rows differ %u, logits rows differ %u, argmax flips %u%s%s", bad_wide, bad_lg, argflip,
                      first.empty() ? "" : "; ", first.c_str());
        check(bad_wide == 0 && bad_lg == 0, what, det);
        char tm[320];
        std::snprintf(tm, sizeof tm,
                      "group step %.2f ms (A+B serial) vs %u one-row steps %.2f ms = x%.2f card time; per row %.2f vs %.2f ms; "
                      "expert misses %.1f vs %.1f a step",
                      t_rows / steps, M, t_solo / steps, t_solo / std::max(1e-9, t_rows), t_rows / steps / M,
                      t_solo / steps / M, double(mr) / steps, double(ms) / steps);
        info("timing M=" + std::to_string(M), tm);
    }
    // whole-state identity per twin pair (both stages)
    uint32_t state_bad = 0;
    std::string sw;
    for (uint32_t l = 0; l < K; ++l) {
        Qwen4ExpModel::SlotState a1, b1, a2, b2;
        std::string e = select(s, l);
        if (e.empty()) e = s.A->stash_slot(a1, pos[l]);
        if (e.empty()) e = s.B->stash_slot(b1, pos[l]);
        if (e.empty()) e = select(s, l + K);
        if (e.empty()) e = s.A->stash_slot(a2, pos[l + K]);
        if (e.empty()) e = s.B->stash_slot(b2, pos[l + K]);
        if (!e.empty()) { check(false, "stash lane " + std::to_string(l), e); return 1; }
        std::string w1, w2;
        const bool okA = same_state(a1, a2, w1), okB = same_state(b1, b2, w2);
        if (!okA || !okB) {
            ++state_bad;
            if (sw.empty()) sw = "lane " + std::to_string(l) + (okA ? "" : " stage A " + w1) + (okB ? "" : " stage B " + w2);
        }
    }
    check(state_bad == 0, "model: every grouped lane's whole state == its one-row twin's (both stages)",
          state_bad ? std::to_string(state_bad) + " lanes differ; first " + sw : std::to_string(K) + " lanes");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool do_self = false, do_ops = false, do_model = false;
    std::string rows_s = "2-16", devs = "0,1", path;
    uint32_t steps = 6, ctx = 4096, long_len = 2046;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](const char* nm) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", nm); std::exit(2); }
            return argv[++i];
        };
        if (a == "--selftest") do_self = true;
        else if (a == "--ops") do_ops = true;
        else if (a == "--model") do_model = true;
        else if (a == "--exact") do_ops = do_model = true;
        else if (a == "--rows") rows_s = val("--rows");
        else if (a == "--steps") steps = uint32_t(std::atoi(val("--steps").c_str()));
        else if (a == "--ctx") ctx = uint32_t(std::atoi(val("--ctx").c_str()));
        else if (a == "--long") long_len = uint32_t(std::atoi(val("--long").c_str()));
        else if (a == "--devices") devs = val("--devices");
        else if (a[0] != '-' && path.empty()) path = a;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (do_self) {
        const int r = selftest();
        std::printf("selftest: %d ok, %d FAIL\n", g_ok, g_fail);
        return r;
    }
    if (path.empty()) {
        std::fprintf(stderr, "usage: ie-q4e-rows-test <model.gguf> [--selftest] [--ops] [--model] [--exact] [--rows 2-16] "
                             "[--steps 6] [--ctx 4096] [--long 2046] [--devices 0,1]\n");
        return 2;
    }
    if (!do_ops && !do_model) do_ops = do_model = true;
    const std::vector<uint32_t> Ms = parse_rows(rows_s);
    uint32_t K = 2;
    for (uint32_t M : Ms) {
        if (M < 2 || M > Qwen4ExpModel::kMaxRows) { std::fprintf(stderr, "--rows: %u outside 2..16\n", M); return 2; }
        K = std::max(K, M);
    }
    const std::vector<uint32_t> dv = parse_rows(devs);
    if (dv.size() != 2) { std::fprintf(stderr, "--devices: two cards (stage A, stage B)\n"); return 2; }
    if (long_len + 300 + 16 * steps * uint32_t(Ms.size()) + 64 > ctx) {
        std::fprintf(stderr, "--ctx %u is too small for --long %u + the steps\n", ctx, long_len);
        return 2;
    }

    GgufReader g;
    if (auto e = g.open(path); !e.empty()) { std::fprintf(stderr, "open: %s\n", e.c_str()); return 1; }
    Qwen4ExpConfig cfg;
    if (auto e = read_qwen4exp_config(g, cfg); !e.empty()) { std::fprintf(stderr, "config: %s\n", e.c_str()); return 1; }
    Tokenizer tok;
    if (auto e = tok.load_from_gguf(g); !e.empty()) { std::fprintf(stderr, "tok: %s\n", e.c_str()); return 1; }
    DeviceAllocator a0, a1;
    if (auto e = a0.init("B70", dv[0]); !e.empty()) { std::fprintf(stderr, "gpu %u: %s\n", dv[0], e.c_str()); return 1; }
    if (auto e = a1.init("B70", dv[1]); !e.empty()) { std::fprintf(stderr, "gpu %u: %s\n", dv[1], e.c_str()); return 1; }
    Qwen4ExpModel A, B;
    const uint32_t hi = cfg.n_layers / 2;
    const uint32_t lanes = do_model ? 2 * K : 2;   // twins for --model; --ops needs the rows buffers only
    std::printf("loading %s: stage A [0, %u) on card %u, stage B [%u, %u) on card %u, %u lanes at ctx %u\n", path.c_str(), hi, dv[0],
                hi, cfg.n_layers, dv[1], lanes, ctx);
    if (auto e = A.load(a0, g, cfg, 0, 0, hi); !e.empty()) { std::fprintf(stderr, "A: %s\n", e.c_str()); return 1; }
    if (auto e = A.init_runtime(ctx, 2048, lanes, ctx); !e.empty()) { std::fprintf(stderr, "A runtime: %s\n", e.c_str()); return 1; }
    if (auto e = B.load(a1, g, cfg, 0, hi, cfg.n_layers); !e.empty()) { std::fprintf(stderr, "B: %s\n", e.c_str()); return 1; }
    if (auto e = B.init_runtime(ctx, 2048, lanes, ctx); !e.empty()) { std::fprintf(stderr, "B runtime: %s\n", e.c_str()); return 1; }
    const std::string whyA = A.rows_off_reason(), whyB = B.rows_off_reason();
    std::printf("rows: stage A %s (rows_max %u, %u slots/layer), stage B %s (rows_max %u, %u slots/layer)\n",
                whyA.empty() ? "ON" : ("OFF: " + whyA).c_str(), A.rows_max(), A.ecache_slots(),
                whyB.empty() ? "ON" : ("OFF: " + whyB).c_str(), B.rows_max(), B.ecache_slots());

    if (do_ops) {
        ops_stage("stage A", A, cfg, 0, 3, cfg.has_ple() ? cfg.ple_layers[0] : -1, false, Ms);
        ops_stage("stage B", B, cfg, hi, hi + 3, -1, true, Ms);
    }
    if (do_model) {
        if (!whyA.empty() || !whyB.empty()) check(false, "model: rows need both stages ON", whyA + " / " + whyB);
        else if (K > std::min(A.rows_max(), B.rows_max()))
            check(false, "model: rows up to " + std::to_string(K), "the caches hold " + std::to_string(std::min(A.rows_max(), B.rows_max())));
        else {
            Stages s{&A, &B};
            model_test(s, cfg, tok, Ms, K, steps, long_len);
        }
    }
    std::printf("\n%d ok, %d FAIL -> %s\n", g_ok, g_fail, g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
