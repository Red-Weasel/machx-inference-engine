# Laguna-shaped attention goldens, from the SAME eager_attention_forward the
# Inkling gate uses (upstream Laguna uses the transformers-standard eager path
# with sinks/bias absent for this build). Purpose: verify the claim in
# docs/laguna/00_PORT_PLAN.md that `ink_attention` is directly reusable for
# Laguna — same signature, bias argument null, Laguna's real GQA shapes.
import struct, sys
import torch, torch.nn as nn
torch.manual_seed(101)
def repeat_kv(hs, n):
    b, k, s, d = hs.shape
    return hs if n == 1 else hs[:, :, None].expand(b, k, n, s, d).reshape(b, k*n, s, d)
out = open(sys.argv[1], "wb")
# Laguna's REAL per-layer geometry: 48 or 72 query heads, 8 KV heads (n_rep 6/9),
# head_dim 128, sliding_window 512 on the sliding layers.
cases = [(4, 48, 8, 128, 16), (4, 72, 8, 128, 16), (8, 48, 8, 128, 64), (2, 72, 8, 128, 700)]
out.write(struct.pack("<I", len(cases)))
for (T, H, KVH, D, KV) in cases:
    q = torch.randn(1,H,T,D); k = torch.randn(1,KVH,KV,D); v = torch.randn(1,KVH,KV,D)
    scaling = D ** -0.5
    # causal + 512-wide sliding window, the shape Laguna's sliding layers use
    mask = torch.zeros(1,1,T,KV)
    for t in range(T):
        for j in range(KV):
            if j > t + KV - T or (t + KV - T) - j >= 512:
                mask[0,0,t,j] = float("-inf")
        mask[0,0,t, max(0, min(KV-1, t + KV - T))] = 0.0   # always one live column
    ks = repeat_kv(k, H//KVH); vs = repeat_kv(v, H//KVH)
    w = torch.matmul(q, ks.transpose(2,3)) * scaling + mask
    w = nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
    o = torch.matmul(w, vs).transpose(1,2).contiguous()
    out.write(struct.pack("<IIIII", T,H,KVH,D,KV))
    out.write(q.transpose(1,2).contiguous().numpy().astype("<f4").tobytes())
    out.write(k.transpose(1,2).contiguous().numpy().astype("<f4").tobytes())
    out.write(v.transpose(1,2).contiguous().numpy().astype("<f4").tobytes())
    out.write(mask[0,0].contiguous().numpy().astype("<f4").tobytes())
    out.write(o.contiguous().numpy().astype("<f4").tobytes())
out.close()
print("golden written:", sys.argv[1])
