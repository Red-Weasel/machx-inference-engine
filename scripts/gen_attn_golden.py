# Golden vectors for ink_attention, from eager_attention_forward
# (modeling_inkling.py:157-181) + repeat_kv (:145), verbatim.
import struct, sys
import torch, torch.nn as nn
torch.manual_seed(31)
def repeat_kv(hs, n_rep):
    b, nkvh, s, d = hs.shape
    if n_rep == 1: return hs
    return hs[:, :, None, :, :].expand(b, nkvh, n_rep, s, d).reshape(b, nkvh*n_rep, s, d)
out = open(sys.argv[1], "wb")
# (T, n_heads, n_kv_heads, head_dim, n_kv, use_bias, use_mask)
cases = [(4,4,2,8,6,1,1), (8,32,8,128,16,1,1), (3,2,2,16,5,0,1),
         (6,8,1,32,10,1,0), (16,4,4,64,24,1,1)]
out.write(struct.pack("<I", len(cases)))
for (T,H,KVH,D,KV,ub,um) in cases:
    q = torch.randn(1,H,T,D); k = torch.randn(1,KVH,KV,D); v = torch.randn(1,KVH,KV,D)
    scaling = D ** -0.5
    bias = torch.randn(1,H,T,KV) * 0.3 if ub else None
    if um:
        mask = torch.zeros(1,1,T,KV)
        m = torch.rand(T,KV) < 0.4
        m[:, 0] = False                     # keep at least one live column
        mask[0,0][m] = float("-inf")
    else:
        mask = None
    # ---- reference ----
    ks = repeat_kv(k, H//KVH); vs = repeat_kv(v, H//KVH)
    w = torch.matmul(q, ks.transpose(2,3)) * scaling
    if bias is not None: w = w + bias
    if mask is not None: w = w + mask
    w = nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
    o = torch.matmul(w, vs).transpose(1,2).contiguous()      # [1,T,H,D]
    # ---- /reference ----
    out.write(struct.pack("<IIIIIII", T,H,KVH,D,KV,ub,um))
    out.write(q.transpose(1,2).contiguous().numpy().astype("<f4").tobytes())   # [T,H,D]
    out.write(k.transpose(1,2).contiguous().numpy().astype("<f4").tobytes())   # [KV,KVH,D]
    out.write(v.transpose(1,2).contiguous().numpy().astype("<f4").tobytes())
    if ub: out.write(bias.contiguous().numpy().astype("<f4").tobytes())        # [H,T,KV]
    if um: out.write(mask[0,0].contiguous().numpy().astype("<f4").tobytes())   # [T,KV]
    out.write(o.contiguous().numpy().astype("<f4").tobytes())
out.close()
print("golden written:", sys.argv[1])
