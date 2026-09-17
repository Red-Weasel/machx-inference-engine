# Golden vectors for lag_attn_gate, from LagunaAttention.forward's gating block
# (upstream modeling_laguna.py) verbatim: softplus in fp32, per-head, broadcast
# across head_dim, applied BEFORE o_proj.
import struct, sys
import torch, torch.nn.functional as F
torch.manual_seed(59)
out = open(sys.argv[1], "wb")
# (T, n_heads, head_dim) — the real per-layer head counts are 48 and 72.
cases = [(4, 8, 16), (2, 48, 128), (3, 72, 128), (1, 1, 4), (16, 48, 128)]
out.write(struct.pack("<I", len(cases)))
for (T, H, D) in cases:
    attn = torch.randn(T, H * D, dtype=torch.float32)
    # Include extreme logits: softplus must not overflow (naive log(1+exp(x))
    # is inf past ~88) and must stay exact for very negative x.
    gl = torch.randn(T, H, dtype=torch.float32) * 3.0
    if gl.numel() >= 4:
        flat = gl.view(-1)
        flat[0], flat[1], flat[2], flat[3] = 120.0, -120.0, 0.0, 88.5
    # ---- reference ----
    gate = F.softplus(gl.float()).to(attn.dtype)
    ref = (attn.view(T, H, D) * gate.unsqueeze(-1)).view(T, H * D)
    # ---- /reference ----
    out.write(struct.pack("<III", T, H, D))
    out.write(attn.contiguous().numpy().astype("<f4").tobytes())
    out.write(gl.contiguous().numpy().astype("<f4").tobytes())
    out.write(ref.contiguous().numpy().astype("<f4").tobytes())
out.close()
print("golden written:", sys.argv[1])
