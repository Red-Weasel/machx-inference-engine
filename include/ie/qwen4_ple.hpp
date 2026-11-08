// include/ie/qwen4_ple.hpp — Qwen3.8-Flash-Next (qwen4exp) PLE n-gram embeddings.
//
// Spec: docs/qwen4/12_ple_ngram.md. Three pieces:
//   1. Host n-gram hash (uint64, EOS-segmented 3-token context) -> 16 table rows
//      per token position.
//   2. Host row gather + IQ4_NL dequant (table never lives in VRAM) -> E [T,2560].
//   3. Device blk.1 PLE layer math AFTER the ple_key/ple_value projections
//      (which run on the engine's existing GEMV kernels).
//
// Conventions follow include/ie/ops.hpp: device pointers are USM, kernels
// return a sycl::event, row-major last-dim-contiguous layout.

#pragma once

#include <sycl/sycl.hpp>

#include "ie/dtype.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ie {

// Fixed qwen4exp PLE geometry (GGUF-verified, spec §1).
inline constexpr uint32_t kPleHeads     = 16;     // 8 bigram + 8 trigram
inline constexpr uint32_t kPleHeadDim   = 160;    // embedding_length_per_layer_input
inline constexpr uint32_t kPleD         = 2560;   // 16*160: the n-gram embedding E's width (HF ple_embed_dim) -- and
                                                  // Flash-Next's hidden, which the layer math takes as a parameter
inline constexpr uint32_t kPleStreams   = 4;      // hc wide-residual streams
inline constexpr uint32_t kPleSI        = 10240;  // 4*2560: Flash-Next's flat group-norm width (kPleStreams * hidden)
inline constexpr uint32_t kPleConvK     = 4;      // ple.conv_kernel
inline constexpr uint32_t kPleConvDil   = 3;      // token-axis dilation
inline constexpr uint32_t kPleStateRows = 9;      // (K-1)*dil conv history columns
inline constexpr uint32_t kPleRowBytes  = 90;     // 160 IQ4_NL elems = 5 blocks * 18 B (the converter's table)
inline constexpr uint32_t kPleRowBytesQ8 = 170;   // 160 Q8_0 elems = 5 blocks * 34 B (community GGUFs: Unsloth Q8_0, UD-Q6_K_XL)

// The table dtypes the gather reads, and a row's bytes for one (0 = not a table dtype). 2026-11-02 (GitHub issue 3): the
// gather always read the table as IQ4_NL, so a community file's Q8_0 table came out as garbage embeddings and NaN logits.
// 2026-11-07: the community tables add Q5_0 (bartowski, 22 B a block), Q5_1 (Unsloth's swappable table, 24 B), Q4_0
// (ggml-org, 18 B), and the lossless F16 / BF16 / F32 tables; a 160-wide row is 5 blocks of 32 in every 32-block type.
inline constexpr uint32_t qwen4_ple_row_bytes(DType dt) noexcept {
    switch (dt) {
        case DType::kIQ4_NL: return kPleRowBytes;      // 90
        case DType::kQ4_0:   return 5u * 18u;          // 90
        case DType::kQ5_0:   return 5u * 22u;          // 110
        case DType::kQ5_1:   return 5u * 24u;          // 120
        case DType::kQ8_0:   return kPleRowBytesQ8;    // 170
        case DType::kF16:    return 160u * 2u;
        case DType::kBF16:   return 160u * 2u;
        case DType::kF32:    return 160u * 4u;
        default:             return 0u;
    }
}

// Hash constants, passed in by the caller (read from GGUF metadata; the gate
// uses the spec values: M, 16 consecutive primes >= 20000003, exclusive
// prefix-sum offsets, eos = 248044).
struct PleHashConsts {
    uint64_t M[3];              // per-window-position multipliers
    uint64_t V[kPleHeads];      // per-head prime vocab sizes
    uint64_t O[kPleHeads];      // per-head row offsets (exclusive prefix sums)
    uint32_t eos;               // EOS token id (segmentation boundary)
};

// Per-slot PLE state: the effective last-2 token history (already
// EOS-segmented, so a predecessor cut by an EOS is stored AS eos) plus the
// slot's device conv-state handle ([kPleStateRows, kPleSI] f32, not owned).
// save()/rollback() give exact spec-decode rewind of the token history; the
// conv-state rewind is the caller's device-buffer stash, same mechanism as
// the DeltaNet conv/SSM state (spec §4).
struct PleHistory {
    uint32_t h1 = 0, h2 = 0;             // effective t_{p-1}, t_{p-2}
    uint32_t saved_h1 = 0, saved_h2 = 0;
    float*   conv_state = nullptr;       // device [9, 10240] f32 (not owned)

    void reset(uint32_t eos) { h1 = h2 = saved_h1 = saved_h2 = eos; }
    void save()              { saved_h1 = h1; saved_h2 = h2; }
    void rollback()          { h1 = saved_h1; h2 = saved_h2; }
};

// N-gram hash (host, exact uint64 integer math — spec §2).
// Per position p: ctx[0]=t_p, ctx[1]=t_{p-1}, ctx[2]=t_{p-2}, where a
// predecessor at/before the last EOS strictly before p (or before position 0)
// reads as EOS, and once one ctx[s] is EOS all further are EOS. The token's
// own EOS does not cut its own context.
//   mixed_2 = (ctx[0]*M[0]) ^ (ctx[1]*M[1])
//   mixed_3 = mixed_2 ^ (ctx[2]*M[2])
//   rows_out[p*16+h] = mixed_2 % V[h] + O[h]  (h<8)   mixed_3-based (h>=8)
// `hist` supplies the effective predecessors of tokens[0] (reset(eos) at
// sequence start) and is advanced past all T tokens, so chunked calls hash
// identically to one monolithic call.
void qwen4_ple_hash(const PleHashConsts& c,
                    const int32_t* tokens, uint32_t T,
                    PleHistory& hist,
                    uint64_t* rows_out /* [T, 16] */);

// Row gather + dequant (host). `table` is the per_layer_token_embd buffer; `dt` is its dtype, which sets the row layout:
// IQ4_NL (row r at byte offset r*90, the layout the engine's converter writes) or Q8_0 (r*170; community GGUFs).
// E_out[p] = concat of the 16 heads' 160-elem dequants, head index slowest -> [T, 2560] f32. Uses the reference dequants
// (ie::ref::dequant_iq4_nl_buffer / dequant_q8_0_buffer) verbatim. Returns "" or a message naming an unsupported dtype
// (nothing is written then).
std::string qwen4_ple_gather(const void* table, DType dt,
                             const uint64_t* rows /* [T, 16] */, uint32_t T,
                             float* E_out /* [T, 2560] */);

// Device workspace for qwen4_ple_layer, in floats (caller allocates USM), SI = kPleStreams * hidden:
// u [T,SI] + gated [T,SI] + g [T,4] + state_in [9,SI].
inline constexpr size_t qwen4_ple_workspace_floats(uint32_t T, uint32_t hidden = kPleD) {
    return size_t(T) * kPleStreams * hidden * 2 + size_t(T) * kPleStreams
         + size_t(kPleStateRows) * kPleStreams * hidden;
}

// blk.1 PLE layer math AFTER the two projections (spec §3):
//   key  [T, 10240] f16 (ple_key output),  v [T, 2560] f16 (ple_value output),
//   H    [T, 4, 2560] f32 wide residual (t-major, then stream) — updated
//        in place: H += gated + SiLU(conv(GN(gated))).
//   k_hat = GN(key, gamma_k); q_hat = GN(H, gamma_q);
//   s_j = dot(k_hat_j, q_hat_j)/sqrt(2560);  g_j = sigmoid(sign(s)*sqrt(clamp(|s|,1e-6,inf)));
//   gated_j = g_j * v;  u = GN(gated, gamma_c);
//   y[c,p] = sum_k W[k,c] * u[c, p-(3-k)*3]  (depthwise causal, kernel 4, dil 3).
// GN = per-2560-group RMS in fp32 (eps from config), times gamma spanning
// 10240 (gammas pre-folded (1+w) — plain multiply). All gate/norm/conv math fp32.
//
// conv_state is the slot's [9, 10240] f32 history: row s = u's token-column at
// chunk-relative position s-9 (zeroed by the caller at sequence start). It is
// read at chunk start and rewritten at chunk end, so chunked processing is
// exact. gamma_* are [10240] f32, conv_w is [4, 10240] f32. `ws` is a device
// scratch buffer of qwen4_ple_workspace_floats(T) floats. All pointers device USM.
sycl::event qwen4_ple_layer(sycl::queue& q,
                            const sycl::half* key, const sycl::half* v,
                            float* H,
                            const float* gamma_k, const float* gamma_q,
                            const float* gamma_c, const float* conv_w,
                            float* conv_state,
                            float* ws,
                            uint32_t T, float eps,
                            const std::vector<sycl::event>& deps = {},
                            uint32_t hidden = kPleD);
// (2026-11-08: `hidden` is the model's hidden size -- the per-stream group, v's width, and SI = 4 * hidden for key, H,
// the gammas, conv_w and conv_state; the widths above are Flash-Next's 2560 / 10240. E's width (16*160, the
// ple_key / ple_value projections' K) does not depend on it. A Qwen 4 with hidden 5120 read past every buffer here.)

}  // namespace ie
