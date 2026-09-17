# V4.1 port — Phase 9 design: decode on the resident runtime (written BEFORE the build)

**Criteria:** `19_PHASE9_DECODE_CRITERIA_2026-09-12.md` (unchanged). **Golden:**
`golden_decode.py` extended (uncommitted until it has run) to dump the state decode starts
from — `d_pre_window_{0,2,20}`, `d_pre_comp_{2,20}`, `d_pre_idxk_{2,20}`, the compressor's
`kvstate/scorestate`, `d_pre_hashcache` — plus `p_hashes` / `d_hashes` / `s{k}_hashes` as
int64 and, for every step, the per-layer outputs; into `scratchpad/golden/decode2/`.

## What the reference does at one token (model.py:700-786, engram.py:160-175)

| state | prefill (`start_pos = 0`) | decode (`seqlen = 1`, `start_pos = pos`) |
|---|---|---|
| window ring `[win=128, 512]` | rows `[0, T)` (T <= win) else the last `win` rows rotated so row `p` sits at slot `p % win` | slot `pos % win` <- the new RoPE'd kv row; the query attends the WHOLE ring, slots holding positions > pos (never written) masked |
| compressed latents `[T/ratio, 512]` (kv-source) | every full group; RoPE at `j * ratio` | `compress_len = (pos + 1) // ratio`; a latent is emitted only when `(pos + 1) % ratio == 0`, RoPE'd at `pos + 1 - ratio`; ratio 2 holds the odd position's `kv`/`score` in `kv_state`/`score_state` |
| index keys (kv-source) | from each latent BEFORE its RoPE, then RoPE'd at the group position | same cadence as the latent |
| `shared_attn.topk_idxs` (index-source) | top-k over `compress_len` keys per query | recomputed for the one query over `compress_len` keys; consumers reuse |
| engram hash cache | compressed ids of every prompt token | the new token's compressed id appended; its 4-gram looks back into the cache |
| RoPE for q / kv | positions `[0, T)` | position `pos` |

The ring is unordered but every stored key carries its own RoPE, and softmax over a set is
permutation-invariant, so the engine may store the ring exactly as the reference does (slot
= `pos % win`) and attend it as one segment with a slot mask: bit-different summation order,
same arithmetic — the fp32 reorder noise is ~1e-7, far under Phase 7's bars.

## The engine's state, per card, per layer (device memory, allocated at the first forward)

```
Ds41LayerState {
  float* win_kv;                 // [win, HD]        the ring, RoPE'd rows, slot = pos % win
  float* comp_kv;  uint32_t nc;  // [max_comp, HD]   kv-source layers: latents (RoPE'd), nc live
  float* idx_k;                  // [max_comp, IHD]  kv-source layers: index keys (RoPE'd)
  float* part_kv, *part_gate;    // [ratio, HD]      ratio-2 sources: the odd position's row + gate
  bool   part_valid;
}
shared per card: int32_t* sh_topk [1, index_topk]; uint32_t sh_nc, sh_topk_w  (as today)
host:   all_ids (the whole sequence so far), n_pos
```

`max_comp = max_tokens / 1` (ratio-1 layers grow one latent per token). Memory per layer:
ring 256 KB, latents at 2048 tokens 4 MiB + keys 1 MiB — nothing that grows with the step
count except the latent/key arrays (criterion 8).

## The forward, generalised: `forward(ids, T, pos0)`

One body, two admitted regimes: `pos0 == 0` with `T` a multiple of every ratio (today's
prefill, which now also FILLS the state), and `T == 1` at `pos0 = n_pos` (decode). Anything
else is refused (chunked prefill is not this phase). Per layer at decode:

1. engram (layers 1, 14): hashes for the new token from `ds41_engram_hash` over the last
   `max_ngram` ids of `all_ids` (the look-back never spans more), gather, `wkv` GEMM at T=1,
   gate.
2. hc mixes / collapse / norm as today (T=1).
3. q: `wq_a` -> norm -> `wq_b` -> RoPE at `pos`.
4. kv: `wkv` -> norm -> RoPE at `pos` -> written to `win_kv[pos % win]`.
5. kv-source layers: `comp_wkv` (and `comp_wgate` at ratio 2) at T=1. Ratio 1: `latent =
   rms(ckv)`; append. Ratio 2: even `pos` -> store the row and gate in `part_*`, no latent;
   odd `pos` -> chunk `[part; new]` through `ds4_compress_pool(n_win=1, rate=2, overlap=false)`
   -> rms -> latent. For a new latent: index key = rms(`idx_wk` · latent) RoPE'd at
   `pos + 1 - ratio`, appended to `idx_k`; the latent RoPE'd at the same position, appended
   to `comp_kv`; `nc += 1`. `sh_nc = nc`.
6. index-source layers: `idx_wq_b` on `qrn`, RoPE at `pos`; `idx_weights`; `ds4_indexer_score`
   over `nc` keys; `ds4_indexer_topk(T=1, positions=[pos], n_keys=nc, ratio)` -> `sh_topk`,
   `sh_topk_w = min(index_topk, nc)`.
7. mask `[1, win + nc]`: ring slot `s` open iff `s <= pos` (all open once `pos >= win-1`);
   compressed part from `ds4_block_bias_topk`. Built on the host at T=1 (2 KB), copied once.
8. `ds4_attention_segs(T=1, a = win_kv (n_a = win), b = comp_kv (n_b = nc))`, sinks; conjugate
   RoPE; `wo_a` block-diagonal at T=1; `wo_b`; hc mix.
9. FFN: hc mixes / collapse / norm; `ds4_router_topk` at T=1; `Ds41ExpertTier::moe` at T=1 (the
   Q8 grouped path with M = 1 rows per expert — correct, not fast; decode speed is not this
   phase); shared expert; hc mix; `pre_mix` swap.
10. After layer 39: collapse, norm, head at T=1 -> logits `[1, V]`; `all_ids.push_back(id)` is
    the caller's (the test feeds the golden's next token, greedy from the engine's logits).

At prefill the same body writes the ring (`slot = p % win` for the last `min(T, win)` rows),
all latents / keys, and the partial state only when `T % ratio != 0` — refused this phase, so
`part_valid = false` after every prefill.

## How the criteria are met

1. State at the boundary: after the prefill, `win_kv` / `comp_kv` / `idx_k` of layers 0, 2, 20
   read back and compared with `d_pre_*` at 3e-3 (the ring compared slot by slot; empty slots
   ignored). 2. Step 0 per layer vs `d_layer_out_{L}` at Phase 7's bars, plus the engram outs.
3. Step-0 token and top-5 set vs `d_meta`. 4. Four greedy steps vs `all_ids`, with the margin
rule. 5. The negative control: step 0 with `pos` off by one must FAIL criterion 2 (a flag on
the test entry). 6. Step hashes vs `d_hashes` / `s{k}_hashes` as integers. 7. The forward
test unchanged. 8. Per-step device allocations are the routed experts' transient bank and
nothing else; the test reads free VRAM before and after the four steps.

## Not in this phase

Decode speed (the GEMV path, prefetch), chunked prefill at `pos0 > 0`, T > 1 at `pos0 > 0`,
sampling, stop sequences, the serving loop, tokenisation inside the engine.
