# Golden vectors for lag_rope_apply, from apply_rotary_pos_emb + rotate_half
# (upstream modeling_laguna.py:263-306) verbatim, at Laguna's real per-layer
# rotary dims (64 on full-attention layers, 128 on sliding).
import struct, sys, torch
torch.manual_seed(83)
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]; x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
out = open(sys.argv[1], "wb")
# (T, n_heads, head_dim, rope_dim)
cases = [(4, 8, 128, 64), (3, 48, 128, 64), (2, 72, 128, 128), (1, 2, 16, 8), (5, 4, 64, 32)]
out.write(struct.pack("<I", len(cases)))
for (T, H, D, RD) in cases:
    x = torch.randn(1, H, T, D, dtype=torch.float32)
    cos = torch.randn(1, T, RD, dtype=torch.float32)
    sin = torch.randn(1, T, RD, dtype=torch.float32)
    # ---- reference ----
    c = cos.unsqueeze(1); s = sin.unsqueeze(1)
    x_rot, x_pass = x[..., :RD], x[..., RD:]
    emb = (x_rot * c) + (rotate_half(x_rot) * s)
    ref = torch.cat([emb, x_pass], dim=-1)
    # ---- /reference ----
    out.write(struct.pack("<IIII", T, H, D, RD))
    out.write(x.transpose(1,2).contiguous().numpy().astype("<f4").tobytes())   # [T,H,D]
    out.write(cos[0].contiguous().numpy().astype("<f4").tobytes())
    out.write(sin[0].contiguous().numpy().astype("<f4").tobytes())
    out.write(ref.transpose(1,2).contiguous().numpy().astype("<f4").tobytes())
out.close()
print("golden written:", sys.argv[1])
