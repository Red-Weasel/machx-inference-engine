# Golden vectors for the Inkling MoE BLOCK (layers 2-41), transcribed verbatim
# from InklingMoE.forward (:422-425) over InklingTopkRouter (:356),
# InklingExperts (:315) and InklingSharedExperts (:394).
#
# The block is:  routed_sum(x)  +  shared_sum(x, gammas)
# where BOTH weight sets come from ONE router call whose log-softmax spans the
# k routed experts AND the shared experts together.
import struct, sys
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(303)
out = open(sys.argv[1], "wb")
# (T, H, n_routed, n_shared, k, ffn, route_scale, gscale)
cases = [(5, 32, 16, 2, 4, 24, 8.0, 1.0), (3, 16, 8, 2, 2, 12, 2.5, 1.0)]
out.write(struct.pack("<I", len(cases)))
for (T, H, E, S, K, FF, rscale, gscale) in cases:
    g = lambda *s: torch.randn(*s, dtype=torch.float32) * 0.2
    x = g(T, H)
    w_router = g(E + S, H)
    bias = g(E)
    gate_up = g(E, 2 * FF, H)      # fused gate|up per expert
    down = g(E, H, FF)
    sh_gate, sh_up, sh_down = g(S, FF, H), g(S, FF, H), g(S, H, FF)
    # ---- router (verbatim) ----
    logits = x @ w_router.t()
    scores = logits.sigmoid()
    routed_scores = scores[..., :-S]
    sel = torch.topk(routed_scores + bias, K, dim=-1, sorted=False)[1]
    routed_logits, shared_logits = logits[..., :-S], logits[..., -S:]
    tl = torch.cat([routed_logits.gather(-1, sel), shared_logits], dim=-1)
    lp = F.logsigmoid(tl)
    w = torch.exp(lp - torch.logsumexp(lp, dim=-1, keepdim=True)) * rscale * gscale
    gammas, tw = w[..., -S:].contiguous(), w[..., :K].contiguous()
    # ---- routed experts (verbatim shape of the loop) ----
    routed = torch.zeros_like(x)
    for t in range(T):
        for r in range(K):
            e = int(sel[t, r])
            gu = F.linear(x[t], gate_up[e])
            gt, up = gu.chunk(2, dim=-1)
            hh = F.silu(gt) * up
            routed[t] += F.linear(hh, down[e]) * tw[t, r]
    # ---- shared experts (verbatim) ----
    shared = torch.zeros_like(x)
    for s in range(S):
        gt = x @ sh_gate[s].t(); up = x @ sh_up[s].t()
        act = F.silu(gt) * up * gammas[:, s:s+1]
        shared += act @ sh_down[s].t()
    y = routed + shared
    out.write(struct.pack("<6I2f", T, H, E, S, K, FF, rscale, gscale))
    for t in (x, w_router, bias, gate_up, down, sh_gate, sh_up, sh_down, y):
        out.write(t.contiguous().numpy().astype("<f4").tobytes())
out.close()
print("golden written:", sys.argv[1])
