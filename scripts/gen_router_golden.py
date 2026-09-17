# Golden vectors for ink_moe_router, produced by InklingTopkRouter.forward
# transcribed VERBATIM from modeling_inkling.py:356-377.
# Emits, per case: T,n_routed,n_shared,k, route_scale, global_scale,
# logits[T*(nr+ns)], bias[nr], then the reference mapping as a DENSE
# per-expert weight row [T*nr] (zero where not selected) plus shared[T*ns].
# Dense-by-expert makes the comparison independent of top-k slot order, which
# torch leaves unspecified (sorted=False) and which the downstream sum ignores.
import struct, sys
import torch, torch.nn.functional as F
torch.manual_seed(77)
out = open(sys.argv[1], "wb")
cases = [(4, 16, 2, 6, 8.0, 1.0), (8, 256, 2, 6, 8.0, 1.0),
         (2, 32, 2, 4, 1.0, 0.5), (16, 256, 2, 6, 8.0, 1.3), (1, 8, 2, 2, 2.0, 1.0)]
out.write(struct.pack("<I", len(cases)))
for (T, nr, ns, k, route_scale, gscale) in cases:
    n_tot = nr + ns
    router_logits = torch.randn(T, n_tot, dtype=torch.float32)
    bias = torch.randn(nr, dtype=torch.float32) * 0.2
    # ---- reference, verbatim ----
    scores = router_logits.sigmoid()
    routed_scores = scores[..., :-ns]
    scores_for_choice = routed_scores + bias
    topk_indices = torch.topk(scores_for_choice, k, dim=-1, sorted=False)[1]
    routed_logits = router_logits[..., :-ns]
    shared_logits = router_logits[..., -ns:]
    topk_logits = torch.cat([routed_logits.gather(-1, topk_indices), shared_logits], dim=-1)
    topk_log_probs = F.logsigmoid(topk_logits)
    topk_weights = torch.exp(topk_log_probs - torch.logsumexp(topk_log_probs, dim=-1, keepdim=True))
    topk_weights = topk_weights * route_scale * gscale
    shared_gammas = topk_weights[..., -ns:].contiguous()
    topk_weights = topk_weights[..., :k].contiguous()
    # ---- /reference ----
    dense = torch.zeros(T, nr, dtype=torch.float32)
    dense.scatter_(1, topk_indices, topk_weights)
    out.write(struct.pack("<IIII", T, nr, ns, k))
    out.write(struct.pack("<ff", route_scale, gscale))
    out.write(router_logits.contiguous().numpy().astype("<f4").tobytes())
    out.write(bias.contiguous().numpy().astype("<f4").tobytes())
    out.write(dense.contiguous().numpy().astype("<f4").tobytes())
    out.write(shared_gammas.contiguous().numpy().astype("<f4").tobytes())
out.close()
print("golden written:", sys.argv[1])
