#!/usr/bin/env python3
"""Hy4-flash expert-format lab: candidate ways of storing the routed experts, judged by Tencent's reference itself.

  python format_lab.py --source <Tencent dir> --ids-file ids.txt --ref-logits <oracle>/logits.f32 --cov hy4_cov.npz
                       [--usage stats.npz ...] --trials "plain:8 plain:5 rot:5 rotk:4:64 hot:8:4:0.25" [--threads 14]

The reference implementation runs on the CPU in fp32 with every routed expert REPLACED by what a candidate format would
store, and the next-token distributions are compared with the unmodified reference's (KL, top-1). No engine in the loop,
so there is no fp16 rounding floor: the number is the format's own effect. Trials:

  plain:B            every matrix as B-bit integers with scale and offset per 16 weights of a row (the K-quant shape)
  rot:B              the latent INPUT space rotated to the principal axes of the experts' real inputs (--cov: their second
                     moment per layer; the rotation folds into the shared routed_expert_down_proj for free), then gate / up
                     stored with bits per group of 32 directions by reverse water-filling at an average of B; down as plain:B
  rotk:B:K           rotated as above; the K strongest directions kept at 16 bits (a side channel of K columns), the rest plain:B
  hot:BH:BC:F        the most-used fraction F of a layer's experts (--usage counts) as plain:BH, the others plain:BC
Bits per weight are reported with the scale / offset overhead counted (2 x 6 bits per 16 weights -> + 0.75).
"""
import argparse, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_oracle  # noqa: E402
from transformers.models.hy_v4_flash.modeling_hy_v4_flash import HYV4FlashExperts  # noqa: E402

def affine(w, bits, B=16):
    """B-weight blocks along the input dim: min / max fit with 2^bits levels. bits 0 = zeros; 16 = unchanged."""
    if bits >= 16: return w
    if bits <= 0: return torch.zeros_like(w)
    n, k = w.shape
    b = w.reshape(n, k // B, B); lo = b.amin(-1, keepdim=True); hi = b.amax(-1, keepdim=True)
    st = (hi - lo) / float(2 ** bits - 1)
    return (lo + torch.round((b - lo) / st.clamp_min(1e-30)) * st).reshape(n, k)

def waterfill(lam, avg_bits, group=32, choices=(0, 2, 3, 4, 5, 6, 8, 16)):
    """Bits per group of `group` directions (sorted by variance): b_g = avg + 0.5 log2(lam_g / geometric mean), snapped to
    the available widths, the offset searched so the average lands on avg_bits."""
    g = lam.reshape(-1, group).mean(1).clamp_min(1e-30)
    base = 0.5 * torch.log2(g / torch.exp(torch.log(g).mean()))
    ch = torch.tensor(choices, dtype=torch.float64)
    lo, hi = -8.0, 24.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        b = ch[(torch.abs((base + mid).unsqueeze(1) - ch.unsqueeze(0))).argmin(1)]
        if b.mean() > avg_bits: hi = mid
        else: lo = mid
    return [int(x) for x in ch[(torch.abs((base + lo).unsqueeze(1) - ch.unsqueeze(0))).argmin(1)]]

class Trial:
    def __init__(self, spec, cfg, cov, usage):
        self.spec = spec; p = spec.split(":"); self.kind = p[0]; self.bits_used = 0.0; self.weights = 0
        self.b = int(p[1]); self.cov = cov; self.rot = {}; self.alloc = {}
        if self.kind == "rotk": self.k = int(p[2])
        if self.kind == "hot":
            self.bc = int(p[2]); frac = float(p[3]); self.hot = {}
            for l in range(usage.shape[0]):
                order = np.argsort(-usage[l], kind="stable"); self.hot[l] = set(int(e) for e in order[: int(round(frac * usage.shape[1]))])
    def rotation(self, ord_):
        if ord_ not in self.rot:
            ev, q = torch.linalg.eigh(torch.from_numpy(self.cov[ord_]).double())
            ev, q = ev.flip(0).clamp_min(0), q.flip(1)
            self.rot[ord_] = q.float()
            self.alloc[ord_] = waterfill(ev, self.b) if self.kind == "rot" else None
        return self.rot[ord_], self.alloc[ord_]
    def count(self, n, bits): self.bits_used += n * (bits + (0.75 if 0 < bits < 16 else 0)); self.weights += n
    def expert(self, ord_, e, gu, dn):
        """gate_up [2I, K], down [K, I] fp32 -> what the format stores, in the basis the forward will use."""
        if self.kind == "plain":
            self.count(gu.numel() + dn.numel(), self.b); return affine(gu, self.b), affine(dn, self.b)
        if self.kind == "hot":
            b = self.b if e in self.hot[ord_] else self.bc
            self.count(gu.numel() + dn.numel(), b); return affine(gu, b), affine(dn, b)
        q, alloc = self.rotation(ord_)
        w = gu @ q                                            # columns = principal directions of the inputs, strongest first
        if self.kind == "rotk":
            out = torch.cat((w[:, : self.k], affine(w[:, self.k:], self.b)), 1) if self.k % 16 == 0 else None
            self.count(w.shape[0] * self.k, 16); self.count(w.shape[0] * (w.shape[1] - self.k), self.b)
        else:
            cols = []
            for g, b in enumerate(alloc):
                seg = w[:, g * 32:(g + 1) * 32]; cols.append(affine(seg, b)); self.count(seg.numel(), b)
            out = torch.cat(cols, 1)
        self.count(dn.numel(), self.b)
        return out @ q.T, affine(dn, self.b)                  # back to the original basis: the same function as folding q into D

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True); ap.add_argument("--ids-file", required=True); ap.add_argument("--ref-logits", required=True)
    ap.add_argument("--cov"); ap.add_argument("--usage", nargs="*", default=[]); ap.add_argument("--trials", required=True)
    ap.add_argument("--threads", type=int, default=14)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model, cfg, full, get = real_oracle.load(a.source, None, False)
    head = get("lm_head.weight").float()
    ids = torch.tensor([[int(x) for x in open(a.ids_file).read().split()]], dtype=torch.long)
    ref = torch.from_numpy(np.fromfile(a.ref_logits, dtype=np.float32).reshape(ids.shape[1], -1)).double()
    lref = torch.log_softmax(ref, -1)
    cov = np.load(a.cov)["xtx"] if a.cov else None
    usage = sum(np.load(u)["mass"].astype(np.float64) for u in a.usage) if a.usage else None
    experts = [layer.mlp.experts for layer in model.layers if hasattr(layer.mlp, "experts")]
    ord_of = {id(m): i for i, m in enumerate(experts)}
    pool = ThreadPoolExecutor(max_workers=a.threads)
    state = {"trial": None}
    def forward(self, hidden_states, top_k_index, top_k_weights):
        tr = state["trial"]; o = ord_of[id(self)]
        final = torch.zeros_like(hidden_states)
        mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
        hit = [int(i[0]) for i in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero() if int(i[0]) != self.num_experts]
        def one(e):
            gu, dn = self.gate_up_proj[e].float(), self.down_proj[e].float()
            if tr is not None: gu, dn = tr.expert(o, e, gu, dn)
            top_k_pos, token_idx = torch.where(mask[e])
            cur = self._apply_gate(F.linear(hidden_states[token_idx], gu))
            return token_idx, F.linear(cur, dn) * top_k_weights[token_idx, top_k_pos, None]
        torch.set_num_threads(1)                 # the experts are spread over the pool's threads, one core each
        try:
            for token_idx, cur in pool.map(one, hit): final.index_add_(0, token_idx, cur.to(final.dtype))
        finally:
            torch.set_num_threads(a.threads)
        return final
    HYV4FlashExperts.forward = forward
    print(f"{'trial':>14} {'bits/weight':>11} {'KL mean':>9} {'median':>8} {'worst':>7} {'top-1 equal':>12} {'seconds':>8}")
    for spec in ["none"] + a.trials.split():
        state["trial"] = None if spec == "none" else Trial(spec, cfg, cov, usage)
        t0 = time.time()
        with torch.no_grad():
            ngram = model.build_ngram_tokens(ids, None); mask = torch.ones_like(ids, dtype=torch.bool)
            out = model(input_ids=ids, ngram_tokens=ngram, engram_mask=mask, use_cache=False)
            lg = torch.log_softmax(F.linear(out.last_hidden_state.float(), head)[0].double(), -1)
        kl = (lref.exp() * (lref - lg)).sum(1)
        tr = state["trial"]; bpw = tr.bits_used / tr.weights if tr and tr.weights else 16.0
        print(f"{spec:>14} {bpw:>11.2f} {float(kl.mean()):>9.5f} {float(kl.median()):>8.5f} {float(kl.max()):>7.3f} {int((lref.argmax(1) == lg.argmax(1)).sum()):>5} of {len(kl):<4} {time.time() - t0:>8.0f}", flush=True)

if __name__ == "__main__": main()
