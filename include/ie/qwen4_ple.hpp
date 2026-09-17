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

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ie {

// Fixed qwen4exp PLE geometry (GGUF-verified, spec §1).
inline constexpr uint32_t kPleHeads     = 16;     // 8 bigram + 8 trigram
inline constexpr uint32_t kPleHeadDim   = 160;    // embedding_length_per_layer_input
inline constexpr uint32_t kPleD         = 2560;   // 16*160, one stream's width
inline constexpr uint32_t kPleStreams   = 4;      // hc wide-residual streams
inline constexpr uint32_t kPleSI        = 10240;  // 4*2560 flat group-norm width
inline constexpr uint32_t kPleConvK     = 4;      // ple.conv_kernel
inline constexpr uint32_t kPleConvDil   = 3;      // token-axis dilation
inline constexpr uint32_t kPleStateRows = 9;      // (K-1)*dil conv history columns
inline constexpr uint32_t kPleRowBytes  = 90;     // 160 IQ4_NL elems = 5 blocks * 18 B

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

// Row gather + dequant (host). `table` is the IQ4_NL per_layer_token_embd
// buffer (row r at byte offset r*90); E_out[p] = concat of the 16 heads'
// 160-elem dequants, head index slowest -> [T, 2560] f32. Uses the reference
// dequant (ie::ref::dequant_iq4_nl_buffer) verbatim.
void qwen4_ple_gather(const void* table,
                      const uint64_t* rows /* [T, 16] */, uint32_t T,
                      float* E_out /* [T, 2560] */);

// Device workspace for qwen4_ple_layer, in floats (caller allocates USM):
// u [T,10240] + gated [T,10240] + g [T,4] + state_in [9,10240].
inline constexpr size_t qwen4_ple_workspace_floats(uint32_t T) {
    return size_t(T) * kPleSI * 2 + size_t(T) * kPleStreams
         + size_t(kPleStateRows) * kPleSI;
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
                            const std::vector<sycl::event>& deps = {});

}  // namespace ie
