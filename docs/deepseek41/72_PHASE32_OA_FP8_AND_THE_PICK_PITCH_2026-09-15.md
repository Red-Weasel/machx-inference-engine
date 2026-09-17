# V4.1 — Phase 32: o_a in FP8 (-1.78 ms/token), and the row-identity fault it flushed out

## Criteria (written before the build)

1. `wo_a` served from the checkpoint's own FP8 bytes at decode, T = 1 and T = 2..8, prefill untouched.
2. Decode gate (`ie-ds41-decode-test`) 28/28: every layer within its bar, next token the golden's, top-5 set equal.
3. P2 row identity kept: `multi`, `rollback`, `dspark` all PASS (a T-row step equals T one-row steps bit for bit).
4. Measured, like for like: the profiled decode test's 2k prompt, ms/token and the named-kernel total.

## 1. The change

`wo_a` is stored FP8 + 32x32 E8M0 scales in the checkpoint; the fp16 copy the engine used was a LOAD-TIME DEQUANT,
so serving the FP8 bytes reads the same weight values with half the bytes: **2.68 GB/token -> 1.34 GB** (40 layers x
8 groups x 4096 x 1024). `gemv_f16_rows` was already at 97 % of peak (docs/71), so fewer bytes was the only lever.

The FP8 GEMV grew a block-diagonal form: column `n` reads activation slice `(n / cols_per_group) * K`
(`gemv_fp8_e4m3_f16_grouped` for T = 1, `gemv_fp8_rows_grouped` for 2..8 rows; `cols_per_group == N` is the plain
kernel). The forward's decode site and the DSpark batch site use it; prefill keeps the fp16 block-diagonal bmm on
the cache's on-demand fp16 view. `Ds41Fp8Mat f8_wo_a` joins the FP8 completeness check.

| profiled decode test, the 2k prompt | before | o_a FP8 | final build (+ the pitch fix below) |
|---|---|---|---|
| `gemv_f16_rows` (all f16 row sites) | 7.22 ms/token | 2.46 | 2.46 |
| `o_a` (FP8 grouped) | — | 2.98 | 2.98 |
| the two together | 7.22 | **5.44 (-1.78)** | 5.44 |
| named kernels | 41.8 ms/token | 40.0 | 40.0 |
| wall, steps 4-35 | 91.8 ms/token = 10.90 tok/s | 87.5 = 11.43 | **86.1 = 11.61** |

Decode gate 28/28 in every column (`p24/nc_new.log`, `p24/oa_fp8.log`, `p24/oa_final.log`). The wall difference
between the last two columns is run-to-run noise (min 69.5 vs 70.1), not the fix.

## 2. The fault: `multi` failed 12, and o_a was NOT the cause

Gating criterion 3 found `multi` failing 12 cases with max |diff| ~9.5 and "37 layers differ, first layer 3 ring".
Isolation, one variable at a time:

| arm | result | reading |
|---|---|---|
| `IE_DS41_DENSE_FP8=0` | still 12 | not the FP8 change |
| `IE_DS4_ATTN_PICKPART=0` (stride partition) | still 12, but exactly **T-1 rows** differ per case | row 0 always right |
| `IE_DS41_GATHER=0` (dense scan, new kill switch) | **PASS 38/38** | the Phase 26/30 gathered path |
| the 2,048-token scenario, any arm | PASS | only when NC < index_topk |

The diag confirmed row 0 bit-identical at all 40 layers; layer 2's own state (ring, latents, index keys, nc) was
identical while its OUTPUT for rows >= 1 was not. That narrows it to something ROW-ADDRESSED in the gathered path
that only matters when NC < 512. Two pitch mismatches and one layout assumption, all in the pick list:

1. **`ds4_indexer_topk` packs its rows at `min(index_topk, n_keys)`**, not `index_topk` (`out[t * top_k + j]` with
   `top_k` clamped). `ds4_block_bias_topk` reads it that way -- which is why the dense path was fine -- but
   `ds41_sort_picks_asc` read `picks[t * index_topk + i]`: rows 1.. were sorted from row 0's tail.
2. The kernel addresses a row at `b_picks + t * n_picks`, and the caller passed the COUNT (`sh_topk_w`) as
   `n_picks`, so rows 1.. were read at the wrong offset a second time.
3. The sort put the `-1` sentinels at the FRONT, so a row's valid picks sat at `[n_sent, k)`: their list positions,
   and therefore the contiguous partition each slice owns, moved with k -- and k = min(index_topk, NC) differs
   between a T-row step and the one-row steps it must equal (NC counts this step's own latents).

Fixing 2 and 3 alone left the failure map unchanged (the first fix attempt); 1 was found by asking what else is
row-addressed. Now: the sort READS at pitch k and WRITES the whole `index_topk` pitch, valid picks ascending then
`-1`; the caller passes `n_picks = index_topk`. The Phase 26 sort and the Phase 30 partition had been gated at T = 1
only (decode test, held-out text) -- neither ran `multi`, which is the gate that would have caught this.

Gates on the final build: **multi 38/38, rollback 74/74, dspark 153/153, decode 28/28** (default configuration:
gather on, pick partition on, FP8 dense on).

## 3. A cost this fix introduces, and the next step

With `n_picks` the pitch (512), the contiguous partition gives every slice `512 / SP = 8` list positions. At
NC >= 512 that is what it always was (2k prompt: `ds4_attention` 212.8 -> 212.8 ms over 32 steps, unchanged). At
NC < 512 only the first `n_valid / 8` slices carry picks: the 12-token prompt's `ds4_attention` went 88.6 -> 145.6 ms
over 32 steps (**+1.8 ms/token, at a context where the wall is 166 ms and disk-bound**).

Next, as its own gated step: a RANK-STRIDED partition (`j = s; j < n_picks; j += SP`). With the sentinels at the
end it is T-independent for the same reason the contiguous form now is, spreads a short row over all slices again,
and does the same 8 iterations per slice at 512 picks. It regroups the partials, so it goes through the logits bar
and the four gates like Phase 30 did.

## Kill switches

`IE_DS41_GATHER=0` (dense scan), `IE_DS4_ATTN_PICKPART=0` (stride partition), `IE_DS41_DENSE_FP8=0` (fp16 dense,
o_a included).
