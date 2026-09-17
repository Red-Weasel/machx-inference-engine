"""CPU stand-ins for DeepSeek-V4.1's `kernel.py`, so its own `model.py` can run here.

The shipped reference cannot execute on this box: `inference/kernel.py` is written in tilelang
and targets CUDA, and the torch here is 2.12.0+cpu. Verified, not assumed — `import tilelang`
and `import triton` both fail and `torch.cuda.is_available()` is False.

So this module replaces SIX leaf primitives and nothing else. Every other line of the reference
— Attention, Block, MoE, Gate, Expert, Compressor, Indexer, Transformer, the routing, the
masking, the RoPE — runs as DeepSeek wrote it. That is the whole point: the golden's authority
comes from running their composition, not my reading of it.

What is substituted, and what that costs:

  act_quant / fp4_act_quant  -> IDENTITY. The reference quantises activations to fp8/fp4 before
      every GEMM. The golden deliberately does not: it computes in fp32 so that "correct" is the
      mathematical intent rather than one particular quantisation schedule. Engine output will
      therefore differ from this golden by activation-quantisation error, which is a design
      choice the engine makes for itself (as it already does for V4).
  fp8_gemm / fp4_gemm        -> exact dequantisation + a full-precision matmul. The dequant is
      the one proven bit-exact against the format spec in Phase 2, so only the GEMM precision
      differs from the reference, never the weight values.
  hc_split_sinkhorn          -> transcribed line by line from the tilelang kernel.
  sparse_attn                -> NOT YET (Phase 5c); raises rather than returning something wrong.
"""
import torch
import torch.nn.functional as F

FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def act_quant(x, *args, **kwargs):
    """Identity: the golden keeps activations in full precision. Returns (x, None) so the
    reference's `linear()` can pass a scale through without caring. Accepts and ignores every
    positional/keyword form the reference uses, including the in-place ones
    (`act_quant(kv, block, fmt, dtype, True)`), whose return value it discards."""
    return x.float(), None


def fp4_act_quant(x, *args, **kwargs):
    """Identity, same reasoning. The compressor calls it as
    `fp4_act_quant(latent, 16, True, scale_dtype=e4m3)` -- in place, return ignored."""
    return x.float(), None


def _deq_e8m0(s):
    """E8M0 byte -> fp32. torch can convert this dtype on CPU, unlike float4_e2m1fn_x2."""
    return s.to(torch.float32)


def dequant_fp8(w, scale, block=32):
    """[N, K] e4m3 with an [N/block, K/block] e8m0 grid -> fp32 [N, K]."""
    wf = w.to(torch.float32)
    sf = _deq_e8m0(scale)
    bn = wf.shape[0] // sf.shape[0]
    bk = wf.shape[1] // sf.shape[1]
    return wf * sf.repeat_interleave(bn, 0).repeat_interleave(bk, 1)


def dequant_fp4(w, scale, block=32):
    """[N, K/2] packed e2m1 with an [N, K/32] e8m0 plane -> fp32 [N, K].

    Low nibble is the EVEN element along K; this is the order the shipped checkpoint uses and
    the one Phase 2 verified against the OCP MX spec on all 16 codes.
    """
    b = w.view(torch.uint8)
    n, kh = b.shape
    out = torch.empty(n, kh * 2, dtype=torch.float32)
    lut = FP4_VALUES
    out[:, 0::2] = lut[(b & 0x0F).long()]
    out[:, 1::2] = lut[(b >> 4).long()]
    return out * _deq_e8m0(scale).repeat_interleave(32, dim=1)


def fp8_gemm(x, xs, w, ws, scale_dtype=None, block_size=32):
    return F.linear(x.float(), dequant_fp8(w, ws, block_size))


def fp4_gemm(x, xs, w, ws, scale_dtype=None, act_block_size=32):
    return F.linear(x.float(), dequant_fp4(w, ws))


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """Transcribed from hc_split_sinkhorn_kernel in the shipped kernel.py.

      pre[j]     = sigmoid(m[j]      * scale[0] + base[j]) + eps
      post[j]    = 2 * sigmoid(m[j+hc] * scale[1] + base[j+hc])
      comb[j,k]  = m[j*hc + k + 2*hc] * scale[2] + base[j*hc + k + 2*hc]
      comb        = softmax(comb, -1) + eps ; comb /= comb.sum(-2) + eps
      then (sinkhorn_iters - 1) x (row-normalise, column-normalise), both with + eps
    """
    b, s, _ = mixes.shape
    hc = hc_mult
    m = mixes.reshape(-1, (2 + hc) * hc).float()
    sc, ba = hc_scale.float(), hc_base.float()
    pre = torch.sigmoid(m[:, :hc] * sc[0] + ba[:hc]) + eps
    post = 2.0 * torch.sigmoid(m[:, hc:2 * hc] * sc[1] + ba[hc:2 * hc])
    comb = (m[:, 2 * hc:] * sc[2] + ba[2 * hc:]).reshape(-1, hc, hc)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre.view(b, s, hc), post.view(b, s, hc), comb.view(b, s, hc, hc)


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Transcribed from sparse_attn_kernel in the shipped kernel.py.

      q [b, m, h, d]; kv [b, n, d] -- ONE shared KV head for all h query heads;
      topk_idxs [b, m, topk] with -1 marking an unused slot.

      score[i, j] = (q[i] . kv[idx_j]) * softmax_scale,  -inf where idx_j == -1
      out[i]      = sum_j softmax(score)[j] * kv[idx_j]

    Two details that are easy to get wrong and are load-bearing:

      * `attn_sink[i]` is a learned logit with NO value vector. It enters the DENOMINATOR only
        (`sum_exp[i] += exp(attn_sink[i] - scores_max[i])`), so it lets a head attend to
        "nothing" and shrink its output rather than being forced to distribute weight.
      * the running max starts at a FINITE -1e30, not -inf. The kernel says why: a row whose
        indices are all -1 would give exp(-inf - -inf) = NaN; with a finite floor it yields an
        all-zero row, which is the training kernel's convention.
    """
    b, m, h, d = q.shape
    qf, kvf = q.float(), kv.float()
    sink = attn_sink.float().view(1, h)
    out = torch.empty(b, m, h, d, dtype=torch.float32)
    for bi in range(b):
        idx = topk_idxs[bi].long()                      # [m, topk]
        valid = idx >= 0
        gathered = kvf[bi][idx.clamp(min=0)]            # [m, topk, d]
        s = torch.einsum("mhd,mtd->mht", qf[bi], gathered) * softmax_scale
        s = s.masked_fill(~valid.unsqueeze(1), float("-inf"))
        mx = torch.maximum(s.amax(dim=-1), torch.full((m, h), -1e30))
        e = torch.exp(s - mx.unsqueeze(-1))
        denom = e.sum(-1) + torch.exp(sink - mx)
        out[bi] = torch.einsum("mht,mtd->mhd", e, gathered) / denom.unsqueeze(-1)
    return out.to(q.dtype)
