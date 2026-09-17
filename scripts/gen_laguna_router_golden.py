# Golden vectors for lag_moe_router, from LagunaTopKRouter.forward verbatim.
# Dense per-expert weight rows make the comparison independent of top-k slot order.
import struct, sys, torch
torch.manual_seed(151)
out = open(sys.argv[1], "wb")
# (T, n_experts, k, norm_topk_prob)
cases = [(4, 16, 4, 1), (8, 256, 10, 1), (2, 32, 6, 0), (16, 256, 10, 1), (1, 8, 2, 1)]
out.write(struct.pack("<I", len(cases)))
for (T, E, K, norm) in cases:
    logits = torch.randn(T, E, dtype=torch.float32)
    bias = torch.randn(E, dtype=torch.float32) * 0.3
    # ---- reference ----
    routing_scores = torch.sigmoid(logits)
    scores_for_selection = routing_scores + bias
    _, selected = torch.topk(scores_for_selection, K, dim=-1)
    rw = routing_scores.gather(-1, selected)
    if norm:
        rw = rw / rw.sum(dim=-1, keepdim=True)
    # ---- /reference ----
    dense = torch.zeros(T, E, dtype=torch.float32)
    dense.scatter_(1, selected, rw)
    out.write(struct.pack("<IIII", T, E, K, norm))
    out.write(logits.contiguous().numpy().astype("<f4").tobytes())
    out.write(bias.contiguous().numpy().astype("<f4").tobytes())
    out.write(dense.contiguous().numpy().astype("<f4").tobytes())
out.close()
print("golden written:", sys.argv[1])
