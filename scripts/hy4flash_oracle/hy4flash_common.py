"""Shared by the Hy4-flash oracle and pack tools: MXFP4 (the expert tier's format), the tensor names of the release, the
n-gram hash multipliers. Needs torch + numpy (the reference venv)."""
import re
import numpy as np
import torch

# OCP MX FP4 (E2M1): code 0..7 -> magnitude, sign in bit 3. One E8M0 exponent byte per 32 consecutive elements of a row:
# weight = e2m1(nibble) * 2^(byte - 127). Element 2i sits in the LOW nibble of byte i, 2i+1 in the high one (the
# compressed-tensors / MiMo layout the engine's Ds41ExpertTier reads).
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MIDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])

def mxfp4_quantize(w: torch.Tensor):
    """w [N, K] float32, K a multiple of 32 -> (nibbles uint8 [N, K/2], exps uint8 [N, K/32], dequantised float32 [N, K]).
    Per block: the exponent of {e0 + 1, e0, e0 - 1} (e0 = the smallest that holds the block's largest magnitude) with the
    least squared error, the smaller exponent on a tie -- so a block that is already on the grid comes back unchanged."""
    N, K = w.shape
    assert K % 32 == 0, K
    b = w.reshape(N, K // 32, 32).float()
    amax = b.abs().amax(-1)
    e0 = torch.ceil(torch.log2(amax.clamp_min(1e-30) / 6.0))
    best_err = best_e = best_idx = None
    for de in (-1.0, 0.0, 1.0):
        e = (e0 + de).clamp(-127.0, 127.0)
        s = torch.exp2(e).unsqueeze(-1)
        x = b / s
        idx = torch.bucketize(x.abs().clamp_max(6.0), E2M1_MIDS)
        err = ((E2M1[idx] * torch.sign(x) * s - b) ** 2).sum(-1)
        if best_err is None:
            best_err, best_e, best_idx = err, e, idx
        else:
            take = err < best_err
            best_err = torch.where(take, err, best_err); best_e = torch.where(take, e, best_e)
            best_idx = torch.where(take.unsqueeze(-1), idx, best_idx)
    s = torch.exp2(best_e).unsqueeze(-1)
    neg = (b < 0) & (best_idx > 0)
    deq = (E2M1[best_idx] * torch.where(neg, -1.0, 1.0) * s).reshape(N, K)
    code = (best_idx.to(torch.uint8) | (neg.to(torch.uint8) << 3)).reshape(N, K // 2, 2)
    nib = code[..., 0] | (code[..., 1] << 4)
    exps = (best_e + 127.0).to(torch.uint8)
    return nib.contiguous(), exps.contiguous(), deq

def mxfp4_dequantize(nib: torch.Tensor, exps: torch.Tensor) -> torch.Tensor:
    N, K2 = nib.shape
    code = torch.stack((nib & 0xF, nib >> 4), dim=-1).reshape(N, K2 * 2).long()
    mag = E2M1[code & 7] * torch.where((code & 8) != 0, -1.0, 1.0)
    return (mag.reshape(N, -1, 32) * torch.exp2(exps.float() - 127.0).unsqueeze(-1)).reshape(N, K2 * 2)

# The reference module's names -> the released checkpoint's (found 2026-11-18 by `mini_model.py keys`).
_RENAMES = [
    (r"^layers\.(\d+)\.attn_hc\.(fn|base|scale)$", r"layers.\1.hc_attn_layer.hc_pre.hc_\2"),
    (r"^layers\.(\d+)\.ffn_hc\.(fn|base|scale)$", r"layers.\1.hc_mlp_layer.hc_pre.hc_\2"),
    (r"^hc_head\.hc_(fn|base|scale)$", r"hc_head.hc_head_\1"),
    (r"^layers\.(\d+)\.self_attn\.gate_proj\.weight$", r"layers.\1.self_attn.linear_gate.weight"),
    (r"^layers\.(\d+)\.self_attn\.sinks$", r"layers.\1.self_attn.learnable_sink_param"),
]
def release_name(ref_name: str) -> str:
    """A HYV4FlashTextModel state-dict name -> the name the released shards use ('model.' prefix; lm_head stays bare)."""
    if ref_name == "lm_head.weight": return ref_name
    for pat, rep in _RENAMES:
        n = re.sub(pat, rep, ref_name)
        if n != ref_name: return "model." + n
    return "model." + ref_name

def hash_multipliers(seed: int, layer_id: int, max_ngram: int, table_size: int):
    """HYV4FlashEngramLayer.draw_hash_multipliers: numpy's default generator seeded per n-gram layer."""
    half_bound = max(1, np.iinfo(np.int64).max // table_size // 2)
    drawn = np.random.default_rng(seed + 10007 * layer_id).integers(0, half_bound, size=max_ngram, dtype=np.int64)
    return [int(v) for v in (drawn * 2 + 1)]
