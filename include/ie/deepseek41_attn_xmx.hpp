// include/ie/deepseek41_attn_xmx.hpp -- Phase 34 (docs/deepseek41/74): XMX gathered flash-decoding for the V4.1
// decode attention. Three launches per call: GATHER the row's live columns (the window slots in slot order, then
// the indexer's picks ascending) into a fixed fp16 axis with a compact mask; a split-K XMX PARTIAL, one work-group
// per (row, 32-head tile, 64-key block), writing (m, l, O); a COMBINE per (row, tile) with the sinks and 1/l in the
// fp32 kernel's order. Rows never share a work-group, so a T-row step equals its one-row steps (P2).
// NOT bit-identical to ds4_attention_segs (fp16 operands, fp32 accumulation): gated by the decode test's bars.
#pragma once
#include "ie/deepseek4_attn.hpp"

#include <sycl/sycl.hpp>
#include <cstdint>
#include <vector>

namespace ie {

// head_dim must be 512 and n_heads a multiple of 32; when kv.n_b > 0, kv.b_picks (pitch kv.n_picks, sentinels
// negative) must be set. The workspace is per queue, grown on demand.
bool ds41_attention_decode_xmx_eligible(const sycl::device& dev, uint32_t T, uint32_t n_heads, uint32_t head_dim);
sycl::event ds41_attention_decode_xmx(sycl::queue& q,
                                      const float* q_in, const Ds4KvSegs& kv,
                                      const float* mask, const float* sinks,
                                      float* y,
                                      uint32_t T, uint32_t n_heads, uint32_t head_dim,
                                      float scaling,
                                      const std::vector<sycl::event>& deps = {});

// Phase 44 (docs/deepseek41/84): a PREFILL CONTINUATION strip's attention over its live keys only. Row t (absolute
// row r = row0 + t of the chunk) sees at most `win` keys of the dense window segment -- `wkeys` [nkw, head_dim] fp32,
// key k at position pos0 - nprev + k, contiguous from k0 = r + nprev - (win - 1) -- with the mask `mwin`
// [chunk rows, nkw], and the indexer's picks into the latent cache `ckv` [nc, head_dim] fp32: `picks`
// [T, pitch] (a negative sentinel is masked), each a 0-bias key exactly as ds4_block_bias_topk would open it. Both
// are GATHERED per row into a (win + pitch) axis padded to 64-key blocks, and ds4_attention_xmx_segs runs its
// prefill kernel over each row's own axis: O(win + pitch) per row instead of O(nkw + nc). picks = null (pitch 0)
// for a window-only layer. NOT bit-identical to the contiguous walk (other block boundaries, so other online-softmax
// rounding): gated by the cont test, real text and perplexity.
sycl::event ds41_attention_cont_xmx(sycl::queue& q, const float* q_in,
                                    const float* wkeys, uint32_t nkw, uint32_t nprev, uint32_t win, const float* mwin, uint32_t row0,
                                    const float* ckv, uint32_t nc, const int32_t* picks, uint32_t pitch,
                                    const float* sinks, float* y,
                                    uint32_t T, uint32_t n_heads, uint32_t head_dim, float scaling);

}  // namespace ie
