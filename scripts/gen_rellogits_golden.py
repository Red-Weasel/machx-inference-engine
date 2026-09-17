# Golden vectors for ink_rel_logits, produced by the REFERENCE forward itself
# (InklingRelativeLogits.forward transcribed verbatim from
# modeling_inkling.py:131-142). Layout: n_cases, then per case
# T,H,d_rel,rel_extent,n_kv, rs[T*H*d], proj[d*extent], qpos[T], kpos[n_kv],
# bias[H*T*n_kv].
import struct, sys
import torch
torch.manual_seed(23)
out = open(sys.argv[1], "wb")
cases = [(4,2,16,8,6), (16,4,16,1024,32), (8,2,16,512,8), (32,16,16,1024,64), (3,1,16,4,5)]
out.write(struct.pack("<I", len(cases)))
for (T, H, d_rel, rel_extent, n_kv) in cases:
    rs = torch.randn(1, T, H, d_rel, dtype=torch.float32)
    proj = torch.randn(d_rel, rel_extent, dtype=torch.float32) * 0.5
    qpos = torch.arange(T, dtype=torch.long) + 3
    kpos = torch.arange(n_kv, dtype=torch.long)
    # --- reference, verbatim ---
    rel_logits = (rs @ proj).transpose(1, 2)
    distance = (qpos[:, None] - kpos[None, :])[None, None, :, :]
    gather_index = distance.clamp(0, rel_extent - 1).expand(*rel_logits.shape[:2], -1, -1)
    position_bias = rel_logits.gather(-1, gather_index)
    bias = position_bias.masked_fill((distance < 0) | (distance >= rel_extent), 0.0)
    # --- /reference ---
    out.write(struct.pack("<IIIII", T, H, d_rel, rel_extent, n_kv))
    out.write(rs.contiguous().numpy().astype("<f4").tobytes())
    out.write(proj.contiguous().numpy().astype("<f4").tobytes())
    out.write(qpos.numpy().astype("<i4").tobytes())
    out.write(kpos.numpy().astype("<i4").tobytes())
    out.write(bias.contiguous().numpy().astype("<f4").tobytes())
out.close()
print("golden written:", sys.argv[1])
