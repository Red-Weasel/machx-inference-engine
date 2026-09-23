#!/usr/bin/env python3
"""P2 gate (docs/mimo26/00_PORT_PLAN.md): compare the engine's log-probs with llama-perplexity's.

  compare_logprobs.py <llama kld.bin> <engine logprobs.bin> [--max-chunks N]

llama.cpp side: `llama-perplexity -c C --chunks K --kl-divergence-base kld.bin` writes "_logits_", n_ctx (i32),
n_vocab (i32), n_chunk (i32), the tokens (n_chunk * n_ctx i32), then per chunk (n_ctx - 1 - n_ctx/2) records of
nv = 2*((n_vocab+1)//2) + 4 uint16: two floats (scale, min_log_prob) then n_vocab quantised log-probs
log_prob[v] = min_log_prob + scale * q[v] (q == 0 means clipped at max_logit - 16).
engine side: `ie-mimo26-ppl --chunk C --chunks K --logprobs out.bin` writes C, vocab, n_chunks (u32), the tokens
(n_chunks * C i32), then per chunk the same record count of fp16 log-softmax rows over the vocab (exact, no quantisation).

Prints per chunk and overall: mean KL(oracle || engine) in nats and top-1 agreement computed EXACTLY as
`llama-perplexity --kl-divergence` does (base log-probs not renormalised, KL over base log-prob > -16), the mean and
min cosine of the log-prob vectors (over the union of the top-256 tokens of either side), and both PPLs.
"""
import struct
import sys

import numpy as np


def read_llama(path):
    with open(path, "rb") as f:
        magic = f.read(8)
        assert magic == b"_logits_", magic
        n_ctx, n_vocab, n_chunk = struct.unpack("<iii", f.read(12))
        tokens = np.frombuffer(f.read(4 * n_chunk * n_ctx), dtype="<i4").reshape(n_chunk, n_ctx)
        nv = 2 * ((n_vocab + 1) // 2) + 4
        per_chunk = n_ctx - 1 - n_ctx // 2
        recs = np.frombuffer(f.read(2 * nv * per_chunk * n_chunk), dtype="<u2").reshape(n_chunk, per_chunk, nv)
    return n_ctx, n_vocab, n_chunk, tokens, recs


def llama_logprobs(rec, n_vocab):
    # exactly llama-perplexity's own decode (tools/perplexity/perplexity.cpp log_softmax with a base): NO
    # renormalisation -- every token below max_logit - 16 sits at the floor min_log_prob, and the KL below skips it
    scale, min_lp = struct.unpack("<ff", rec[:4].tobytes())
    q = rec[4:4 + n_vocab].astype(np.float64)
    return min_lp + scale * q


def read_engine(path):
    with open(path, "rb") as f:
        C, V, K = struct.unpack("<III", f.read(12))
        tokens = np.frombuffer(f.read(4 * K * C), dtype="<i4").reshape(K, C)
        per_chunk = C - 1 - C // 2
        data = np.frombuffer(f.read(2 * V * per_chunk * K), dtype="<f2").reshape(K, per_chunk, V)
    return C, V, K, tokens, data


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    max_chunks = None
    if "--max-chunks" in sys.argv:
        max_chunks = int(sys.argv[sys.argv.index("--max-chunks") + 1])
    n_ctx, n_vocab, n_chunk, tokens, recs = read_llama(sys.argv[1])
    C, V, K, etok, eng = read_engine(sys.argv[2])
    assert C == n_ctx and V == n_vocab, (C, n_ctx, V, n_vocab)
    K = min(K, n_chunk, max_chunks or K)
    same = int((etok[:K] == tokens[:K]).sum())
    print(f"tokens: {same} of {K * C} identical between the oracle's and the engine's chunks")
    if same != K * C:
        k, j = np.argwhere(etok[:K] != tokens[:K])[0]
        sys.exit(f"TOKEN MISMATCH at chunk {k} position {j}: oracle {tokens[k, j]} engine {etok[k, j]} -- the comparison would be meaningless")
    first = n_ctx // 2
    tot_kl = tot_top1 = tot_cos = 0.0
    tot_n = 0
    min_cos = 1.0
    nll_o = nll_e = 0.0
    for k in range(K):
        kl_sum = top1 = cos_sum = 0.0
        for i in range(recs.shape[1]):
            lo = llama_logprobs(recs[k, i], n_vocab)
            le = eng[k, i].astype(np.float64)
            le = le - np.logaddexp.reduce(le)
            keep = lo > -16.0                        # llama.cpp's KL skips the base's clipped tail
            kl = float(np.sum(np.exp(lo[keep]) * (lo[keep] - le[keep])))
            kl_sum += kl
            top1 += int(np.argmax(lo) == np.argmax(le))
            idx = np.union1d(np.argpartition(lo, -256)[-256:], np.argpartition(le, -256)[-256:])
            a, b = lo[idx], le[idx]
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            cos_sum += cos
            min_cos = min(min_cos, cos)
            tgt = tokens[k, first + i + 1]
            nll_o -= lo[tgt]
            nll_e -= le[tgt]
        n = recs.shape[1]
        print(f"chunk {k}: mean KL {kl_sum / n:.5f} nats, top-1 {100 * top1 / n:.2f}%, mean cos {cos_sum / n:.5f}")
        tot_kl += kl_sum; tot_top1 += top1; tot_cos += cos_sum; tot_n += n
    print(f"OVERALL over {tot_n} positions: mean KL(oracle||engine) {tot_kl / tot_n:.5f} nats, top-1 {100 * tot_top1 / tot_n:.2f}%, "
          f"mean cos {tot_cos / tot_n:.5f}, min cos {min_cos:.5f}, PPL oracle {np.exp(nll_o / tot_n):.4f} engine {np.exp(nll_e / tot_n):.4f} "
          f"({100 * (np.exp(nll_e / tot_n) / np.exp(nll_o / tot_n) - 1):+.3f}%)")


if __name__ == "__main__":
    main()
