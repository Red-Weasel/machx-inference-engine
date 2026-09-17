# Golden vectors for a WHOLE Inkling decoder layer (dense variant, i.e. blk.0/1),
# transcribed VERBATIM from modeling_inkling.py: InklingDecoderLayer.forward
# (:567) over InklingAttention.forward (:217), InklingShortConvolution (:501),
# InklingRMSNorm (:99) and InklingMLP.
# Purpose: gate the ASSEMBLY (op order, the four shortconv sites, where each
# residual attaches) — which is where port bugs live once the ops are correct.
import struct, sys
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(202)

def rms(x, w, eps=1e-6):
    v = x.float()
    return (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w

def sconv(x, w, K):           # [T,C] , w [C,K] -> InklingShortConvolution
    T, C = x.shape
    residual = x
    hs = x.transpose(0,1).unsqueeze(0)                    # [1,C,T]
    conv = F.conv1d(hs, weight=w.unsqueeze(1), bias=None, padding=K-1, groups=C)[:,:,:T]
    return conv[0].transpose(0,1) + residual

out = open(sys.argv[1], "wb")
cases = [(6, 64, 4, 2, 16, 8, 4, 128, 12),   # T,H,heads,kvh,hd,d_rel,K,extent,ffn
         (4, 32, 2, 2, 16, 8, 4, 64, 8)]
out.write(struct.pack("<I", len(cases)))
for (T, H, NH, KVH, HD, DR, K, EXT, FF) in cases:
    g = lambda *s: torch.randn(*s, dtype=torch.float32) * 0.2
    x   = g(T, H)
    w_in, w_post = g(H), g(H)
    wq, wk, wv, wo = g(NH*HD, H), g(KVH*HD, H), g(KVH*HD, H), g(H, NH*HD)
    wr  = g(NH*DR, H)
    qn, kn = g(HD), g(HD)
    sc_k, sc_v, sc_a, sc_m = g(KVH*HD, K), g(KVH*HD, K), g(H, K), g(H, K)
    proj = g(DR, EXT)
    wg, wu, wd = g(FF, H), g(FF, H), g(H, FF)
    # ---------- reference ----------
    residual = x
    h = rms(x, w_in)
    q = h @ wq.t(); k = h @ wk.t(); v = h @ wv.t()
    k = sconv(k, sc_k, K); v = sconv(v, sc_v, K)
    q = rms(q.view(T, NH, HD), qn).view(T, NH, HD)
    k = rms(k.view(T, KVH, HD), kn).view(T, KVH, HD)
    v = v.view(T, KVH, HD)
    rel = (h @ wr.t()).view(T, NH, DR)
    qp = torch.arange(T); kp = torch.arange(T)
    rl = (rel @ proj).transpose(0,1)                       # [NH,T,EXT]
    dist = (qp[:,None] - kp[None,:])[None,:,:]
    gi = dist.clamp(0, EXT-1).expand(NH,-1,-1)
    bias = rl.gather(-1, gi).masked_fill((dist<0)|(dist>=EXT), 0.0)
    nrep = NH // KVH
    ks = k[:,:,None,:].expand(T,KVH,nrep,HD).reshape(T,NH,HD)
    vs = v[:,:,None,:].expand(T,KVH,nrep,HD).reshape(T,NH,HD)
    scores = torch.einsum('thd,shd->hts', q, ks) * (HD ** -0.5) + bias
    causal = torch.full((T,T), float("-inf")).triu(1)
    scores = scores + causal[None]
    p = F.softmax(scores, dim=-1, dtype=torch.float32)
    ao = torch.einsum('hts,shd->thd', p, vs).reshape(T, NH*HD)
    ao = ao @ wo.t()
    ao = sconv(ao, sc_a, K)
    h = residual + ao
    residual = h
    hh = rms(h, w_post)
    mlp = (F.silu(hh @ wg.t()) * (hh @ wu.t())) @ wd.t()
    mlp = sconv(mlp, sc_m, K)
    y = residual + mlp
    # ---------- /reference ----------
    out.write(struct.pack("<9I", T,H,NH,KVH,HD,DR,K,EXT,FF))
    for t in (x,w_in,w_post,wq,wk,wv,wo,wr,qn,kn,sc_k,sc_v,sc_a,sc_m,proj,wg,wu,wd,y):
        out.write(t.contiguous().numpy().astype("<f4").tobytes())
out.close()
print("golden written:", sys.argv[1])
