# V4.1 port — DSpark P1: the Python golden, what it consumes, what it cannot compute, the engine checklist

**Status: the script `tools/ds41_reference/golden_dspark.py` is written and its `--selftest` was run
(tiny random shapes, one core, 1.03 s wall, exit 0). The REAL golden was NOT run** — a GPU gate
evaluator was measuring on this machine, so no model load and no multi-core work was allowed. Every
number about the real run's time and memory below is **Derived** from the recorded timings of the
existing golden scripts, not measured. Pineapples are spiky: treat those as estimates until the run.

Companion to docs/deepseek41/48 (the DSpark design brief; sections A, C.3 and E/P1 are the
contract this golden serves). Line numbers are `~/models/DeepSeek-V4.1-Flash/inference/
model.py` unless a file is named. **Verified** = read in the code, the checkpoint headers, or the
golden directory; **Derived** = arithmetic from a recorded number; **Assumption** = not established.

---

## 1. What the script computes (Verified: it runs the reference's own composition)

The script runs `Transformer.forward_spec` (1275-1282) — `forward_embed` (1128-1135), the three
`DSparkBlock`s (1122-1126 → `Block.forward` 968-994 with `DSparkAttention.forward` 1033-1074 and
the 128-expert / top-3 `MoE`), then `forward_head` (1137-1156: `hc_pre`, the stage-2 `norm`, the
TIED head, the sequential Markov bias + argmax chain 1149-1153, the confidence head 1155) — with the
reference's modules and only the six leaf kernels replaced by `tools/ds41_reference/kernel.py`
(docs/deepseek41/09: `act_quant`/`fp4_act_quant` identity, exact FP8/FP4 dequant + fp32 matmul,
transcribed Sinkhorn and `sparse_attn`). Same weight conventions as `golden_decode.py`: FP8 dense
weights kept FP8 and dequantised exactly per call; `wo_a` dequantised to fp32 up front (the
block-diagonal einsum at 1072 reads `.weight` directly); experts `np.memmap`-backed FP4, only
routed-to slots ever read; `temperature = 0` so `sample` is argmax (1288-1289) — **the reference's
default is 1 (Gumbel-max, 1290-1292); a golden regenerated without `temperature=0` would draft
randomly.**

The intermediates are captured by forward hooks (no reference line is edited), and every pass is
self-checked independently of the module composition:

| check | what it asserts | reference |
|---|---|---|
| index set | every draft row attends to `[0, min(win, p+1))` ++ `[win, win+5)` — all filled ring slots and ALL five draft rows, no causal mask | 1021-1029, 1064 |
| ring slot | slot `p % win` of each stage holds that stage's RoPE'd `kv_norm(wkv(main_x))` at position `p`; at prefill, slots `pos % win` for the last `min(T, win)` positions | 1040-1041, 1044-1052, 1065 |
| head | norm → tied head → Markov bias chain → argmax → confidence, recomputed from the raw weights, must reproduce `logits_base`, `logits`, `output_ids` (exactly) and `confidence` | 1144-1156, 1083-1086, 1095-1097 |
| non-causal control | perturbing the NOISE token's embedding (rows 1-4's input only) moves draft row 0's attention output; the perturbed pass at the same position leaves the ring unchanged (the write is idempotent) | 1131-1133, 1064-1067 |
| shapes / finiteness | every dumped tensor has the engine-facing shape and is finite | — |

Two things the golden also records per position, for the engine test's soft rules: `route_gap`
(the 3rd-minus-4th router score per row, Phase 7's near-tie rule) and `draft_margin` (each draft
row's top-1 minus top-2 of the biased logits, the margin rule of `tools/ds41_decode_test.cpp:273`).

### 1.1 The dumps (`<out>/sp{pos}_*`, plus `sp_prefill_*`, plus `sp_meta.json` with every shape)

Real shapes: `dim` 5120, `hc` 4, `win` 128, `head_dim` 512, `V` 129280, `B` 5 draft rows, `TK` 3,
stages `s ∈ {0,1,2}`. `.f32`/`.i32` are raw little-endian, C order, as every existing golden.

| file | shape | what | reference |
|---|---|---|---|
| `sp_prefill_main_hidden.f32` | [1, 12, 15360] | mean-over-hc of the stream entering layers 37, 38, 39, concatenated, for the 12 prompt positions | 1264-1271 |
| `sp_prefill_main_x.f32` | [1, 12, 5120] | `main_norm(main_proj(main_hidden))` | 1130 |
| `sp_prefill_window_{s}.f32` | [128, 512] | each stage's ring after prefill seeding: slots 0..11 hold RoPE'd `kv_norm(wkv(main_x[p]))`, the rest zero | 1044-1052 |
| `sp{pos}_main_hidden.f32` | [1, 1, 15360] | the forced input at position `pos` | 1264-1271 |
| `sp{pos}_main_x.f32` | [1, 1, 5120] | | 1130 |
| `sp{pos}_embed.f32` | [1, 5, 4, 5120] | row 0 = embed(t_{pos+1}), rows 1-4 = embed(128799), broadcast to hc | 1131-1134 |
| `sp{pos}_window_{s}.f32` | [128, 512] | the ring AFTER this pass's write of slot `pos % 128` | 1065 |
| `sp{pos}_topk_{s}.i32` | [5, pos+1+5] | the index set (ring slots, then 128+0..4) | 1021-1029 |
| `sp{pos}_attn_in_{s}.f32` | [1, 5, 5120] | `attn_norm(hc_pre(h))` — the attention's `x` | 983-984 |
| `sp{pos}_attn_out_{s}.f32` | [1, 5, 5120] | `wo_b` output | 1074 |
| `sp{pos}_ffn_in_{s}.f32` | [1, 5, 5120] | `ffn_norm(hc_pre(h))` — the MoE input | 989-990 |
| `sp{pos}_moe_out_{s}.f32` | [1, 5, 5120] | routed + shared expert output | 889-905 |
| `sp{pos}_layer_out_{s}.f32` | [1, 5, 4, 5120] | the block's output stream | 993 |
| `sp{pos}_ffn_pre_{s}.f32` | [1, 5, 4] | the `ffn_pre` mix the block returns (stage 2's feeds `forward_head`'s `hc_pre`) | 993, 1144 |
| `sp{pos}_route_idx_{s}.i32` / `route_w_{s}.f32` / `route_gap_{s}.f32` | [5, 3] / [5, 3] / [5] | forced routing per row; 3rd-vs-4th margin | 809-827 |
| `sp{pos}_head_in.f32` | [1, 5, 5120] | `hc_pre(h, ffn_pre_2)` — pre-norm; also the confidence head's `hidden` | 1144, 1155 |
| `sp{pos}_logits_base.f32` | [5, 129280] | `head(norm(x))` before any Markov bias | 1145 |
| `sp{pos}_markov_embed.f32` | [5, 256] | `markov_head.embed[output_ids[i]]`, i = 0..4 | 1084 |
| `sp{pos}_markov_bias.f32` | [5, 129280] | `markov_head.head(embed)` per row | 1085 |
| `sp{pos}_logits.f32` | [5, 129280] | `logits_base + markov_bias` — what `forward_head` returns | 1151 |
| `sp{pos}_output_ids.i32` | [6] | `[t_{pos+1}, d_1 .. d_5]` = positions pos+1 .. pos+6 | 1146-1153 |
| `sp{pos}_confidence.f32` | [5] | raw fp32 scalars (not sigmoided) | 1155, 1097 |

Per position that is 32 files (~8 MB, the three `[5, V]` rows dominate); with 4 positions plus the
prefill set, ~35 MB total.

---

## 2. Which existing dumps it consumes (exact names, all under `ie_golden/decode2/`, all Verified present with the asserted sizes)

| file | size | used as |
|---|---|---|
| `d_meta.json` | — | `prompt_ids` (12), `all_ids` (17), `n_decode` (4), `dim`, `hc_mult`, `n_layers` |
| `p_layer_out_36.f32`, `p_layer_out_37.f32`, `p_layer_out_38.f32` | 983,040 B = [1, 12, 4, 5120] each | the prefill's stream entering layers 37, 38, 39 → `sp_prefill_main_hidden` |
| `d_layer_out_36/37/38.f32` | 81,920 B = [1, 1, 4, 5120] | position 12 (decode step 0) → `sp12_main_hidden` |
| `s1_layer_out_36/37/38.f32`, `s2_…`, `s3_…` | 81,920 B each | positions 13, 14, 15 |
| `d_logits_step1.f32` … `d_logits_step4.f32` | 517,120 B = [129280] | asserted: `argmax == all_ids[pos+1]`, the `input_ids` of the pass at `pos` |

**Why `layer_out_{L-1}` is "the stream entering L" (Verified).** `Transformer.forward` applies the
engram first (1262-1263), THEN appends `h.mean(dim=2)` when `i in target_layer_ids` (1265-1266),
THEN runs the layer (1267). So the capture is post-engram, pre-layer. Layers 37-39 are not engram
layers (`engram_layer_ids = [1, 14]`, config.json), so the stream entering L is exactly the
previous layer's output, which `golden_decode.py` dumps as `{prefix}_layer_out_{L-1}` right after
`blk(...)`. The script codes the general rule (engram layer → `{prefix}_engram_out_{L}`, else
`{prefix}_layer_out_{L-1}`) so a changed `dspark_target_layer_ids` cannot silently pick the wrong
tensor. The three means are concatenated in target order (1271): `[entering 37 | entering 38 |
entering 39]`.

**Which forced step, and why (Verified).** `golden_decode.py` ran the 12-token prompt (positions
0-11), then decode steps at positions 12, 13, 14, 15, dumping the per-layer stream at every step
(`d_`, `s1_`, `s2_`, `s3_`). That is enough for the drafter at **every one of those four
positions**, and the script runs all four in order: the prefill seeding (`forward_spec` at
`start_pos = 0` over the 12 prompt rows, 1044-1052, so slots 0-11 are filled) followed by the pass
at 12 (which writes slot 12, 1065), then 13, 14, 15 — so the ring at position 15 is the
reference's own, built by the reference's own writes, not reconstructed. The position algebra:
the backbone's step `k` fed `all_ids[12+k]` at `start_pos = 12+k` and produced
`d_logits_step{k+1}` → `all_ids[13+k]` (asserted, all four margins 0.35-7.6); the drafter's pass at
`p = 12+k` takes `main_hidden(p)` and `input_ids = all_ids[p+1]`, exactly the smoke test's order
(1305-1309) and generate.py's convention (69-76). Draft `d_j` (j = 1..5) is the prediction for
position `p+1+j`; `all_ids` reaches position 16, so the golden prints, for information only, how
many drafts equal the backbone's own greedy continuation (3 comparable at p = 12, 2 at 13, 1 at
14, 0 at 15). This is not a criterion — the drafter is a different network.

Not consumed: `d_pre_*` / `d_step*_window_*` (backbone rings, irrelevant to the drafter),
`d_route_*` (backbone routing), `d_hashes` / `p_hashes` (engram), `ie_golden/model/*` (the same
prompt through `golden_model.py`; equivalent but not needed).

---

## 3. What it CANNOT compute from the existing goldens — the reduced contract P1 can validate

1. **Only `pos0 ∈ {12, 13, 14, 15}`: a partly filled ring.** docs/48 P1 criterion 1 asks for
   `pos0 < 128`, `pos0 = 128` (exactly filled) and `pos0 = 1000` (wrapped). The wrapped and
   exactly-filled cases need the backbone's stream at layers 36-38 for ≥ 128 (resp. ≥ 1000)
   positions. **No such dump exists** (Verified: `ie_golden/model/` and `decode2/` are the 12-token
   prompt; `pp_ids_2048.i32` / `replay_pp2048_greedy4.i32` are token ids only). Producing them
   means re-running `golden_decode.py` (or a variant that dumps only layers 36-38) with a ≥ 130-
   or ≥ 1000-token prompt; at those lengths the backbone touches most of its 384 experts per layer,
   so the prefill cost is dominated by expert page-in and dequant (12 tokens took 91 s; 1,000
   tokens is **Unknown**, plausibly tens of minutes to an hour — not attempted).
   **Reduced contract for P1:** the stage bars, the discrete `output_ids`, the confidence, the
   forced-routing flips and the prefill-seeding ring check are all validated at the partly filled
   ring (4 positions, ring slots 0..15). The ring's slot algebra at wrap (`slot = pos % 128`, the
   `cutoff = T % win` split at 1048-1051) is exercised by the self-test at window 16 with a
   20-position prefill and passes at pos 20/21 — that validates the reference's seeding *logic* the
   engine must mirror, not the real weights at real positions. A real wrapped golden is a separate,
   costed job (option: a synthetic-`main_hidden` wrapped run — random `[1, 130, 15360]` rows through
   the same script would take ~2 min and check the engine's ring/mask/RoPE algebra at real scale
   with meaningless-but-deterministic numbers; it is not written because the founder should choose
   between that and the real long-prompt regen).
2. **The engine's OWN `main_hidden`.** The golden forces the reference's fp32 `main_hidden`. The
   engine's capture at layers 37-39 inherits the backbone's accumulated error (the card-1 bar
   `bar_of(L ≥ 20) = 3.5e-2`, `ds41_decode_test.cpp:191`). So the P1 test must **force**
   `sp{pos}_main_hidden.f32` into the drafter for the stage bars, and only *report* the engine's
   own capture against it (an informational number under the backbone's bar, not a new criterion).
3. **Temperature > 0.** Argmax only, per docs/48 B.3's recommendation; no speculative-sampling
   golden.
4. **"Five 1-row calls ≠ one 5-row call" (P1 criterion 4).** The golden supplies the perturbation
   control instead (row 0 moves when only rows 1-4's input changes: 4.7e-2 to 8.9e-2 relative at
   tiny scale; the real-scale value lands in `sp_meta.json` as `control_rel` at position 12). The
   engine-side control (a causal mask among the 5 drafts must FAIL `attn_out_0` rows 0-3) is the
   engine test's to run.
5. **Batch > 1, images (`image_mask`), the DSpark scheduler's use of the confidence.** Out of
   scope; drafts are text-only (1126).

---

## 4. The command, and the expected runtime and memory of the real golden (Derived — NOT measured)

```
cd "~/00 - Inference Engine"
OMP_NUM_THREADS=16 ~/venv/bin/python tools/ds41_reference/golden_dspark.py \
    ~/models/DeepSeek-V4.1-Flash/ie_golden/dspark \
    ~/models/DeepSeek-V4.1-Flash \
    ~/models/DeepSeek-V4.1-Flash/ie_golden/decode2 4
```

(`OMP_NUM_THREADS=16` is what `~/ds41_work/golden_regen.sh` used for the existing
goldens; a `run golden_dspark …` line belongs in that script — not added here, it is outside this
task's write set. Launch it from a terminal in the foreground, never `run_in_background`: the
harness's low-memory watchdog has killed background runs on this box while pinned banks were live.)

Basis (Verified, `golden_regen.log`): a 40-layer decode step at 6 experts/layer = 22-25 s
(≈ 240 expert dequants + 40 layers of per-call FP8 dequant of `wq_b` [32768, 1280], `wo_b`
[5120, 8192], `wq_a`, `wkv`, the 3 shared-expert matrices); "all 40 blocks alive" 4 s; the
12-token prefill 91 s.

| phase | estimate | basis |
|---|---|---|
| imports, tokenizer, safetensors handles | ~5 s | as golden_decode |
| `embed.weight` + `head.weight` bf16 → fp32 | ~10-20 s, **5.3 GB** resident | 2 × 1.32 GB read, 2 × 2.65 GB fp32 (golden_decode does the same, untimed) |
| 3 stages' dense binds (3 × ~0.17 GB FP8) + `wo_a` → fp32 (3 × 134 MB) + 384 memmap views | ~3 s | as `load_block` |
| prefill seeding (`main_proj` dequant 315 MB + 3 × `wkv` over 12 rows) | ~2 s | one FP8 [5120, 15360] dequant |
| one pass: 3 stages × (dense dequants + ≤ 15 unique experts × 3 FP4 dequants + shared) | **~6-10 s** | 45/240 of a backbone step's expert work + 3/40 of its dense work, each stage's `wq_b`/`wo_b` dequant ≈ 0.4 s |
| 4 positions + 1 control pass + 2 independent head recomputes per pass | ~35-50 s | 5 passes |
| **total** | **~1-2 min at 16 threads; ~3-5 min single-threaded** | Pineapple: not measured |
| **RAM** | **~7-8 GB** resident (5.3 embed/head + 0.26 Markov + 0.4 `wo_a` + ≤ 0.6 transient dequants) + up to ~4 GB evictable page cache for touched experts | Derived |
| disk | ~35 MB of dumps | §1.1 |

---

## 5. The self-test (Verified: run, exit 0)

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 19 ~/venv/bin/python \
    tools/ds41_reference/golden_dspark.py --selftest <scratch_dir>
  wrapped: prefill 20 > window 16, then 20, 21  pos 20: ids [109, 289, 51, 429, 181, 496] … row-0 control 6.70e-02
  wrapped: …                                    pos 21: … row-0 control 4.71e-02
  partial: prefill 6, then 6, 7                 pos  6: … row-0 control 5.94e-02
  partial: …                                    pos  7: … row-0 control 8.88e-02
selftest OK: 136 dumps … 0.0s on 1 thread        (wall 1.03 s including imports, max RSS 269 MB)
```

Shapes: dim 64, 4 heads (head_dim 32, rope 16), 8 experts / top-2, vocab 512, 2 stages, window 16,
3 target layers (so `main_proj` is `[dim, 3·dim]` like the real one), block 5, Markov rank 8;
random weights from a fixed seed; `default_dtype = float32` so every `Linear` is plain fp32 and the
expert dtype is `None`. It exercises both prefill-seeding branches of 1044-1052 (`T ≤ win` and the
`cutoff = T % win` split), a wrapped and a partly filled ring, and all five self-checks of §1.

What the self-test does **not** exercise (Pineapple): the safetensors loader, the memmap FP4 expert
path, the per-call FP8 dequant path, `bias_vl`, the tokenizer, and the decode2 file plumbing.
Those lines are `golden_decode.py`'s, copied verbatim, and that script produced the goldens the
engine passes against — but "copied" is not "run". The decode2 files' sizes were checked
separately (§2) and `py_compile` is clean. One real-model bug class the self-test *did* catch and
the script now handles: `Attention.__init__` builds `wo_a` in bf16 regardless of `default_dtype`
(645-649), and the einsum at 1072 needs it to match the fp32 stream — the real run converts it
through `put(…, "fp8->f32")` exactly as `golden_decode.py` does.

---

## 6. The engine-side P1 checklist

### 6.1 The six binds (Verified against the shard headers; dtypes/shapes as the checkpoint stores them)

Today they are swallowed by the `mtp_extra` name-claim at `src/model/deepseek41.cpp:365-371`,
which exists only so `unclaimed()` stays empty. P1 deletes those clauses and binds by role — a
`Ds41Dspark` struct next to `embed, lm_head, final_norm` (`include/ie/deepseek41.hpp:158`), six
`b.bind` calls in the loader (`deepseek41.cpp:249-253` is the per-layer pattern; the fp8 one takes
the `32, 32` block args like `w.wq_a` does):

| tensor | dtype | shape | consumer |
|---|---|---|---|
| `mtp.0.main_proj.weight` + `.scale` | F8_E4M3 + F8_E8M0 | [5120, 15360] + [160, 480] (32×32 blocks) | `main_x = main_norm(main_proj(main_hidden))`, once per pass, T rows at prefill seeding |
| `mtp.0.main_norm.weight` | BF16 | [5120] | RMSNorm after `main_proj` |
| `mtp.2.norm.weight` | BF16 | [5120] | the drafter's final norm (NOT `norm.weight`) |
| `mtp.2.markov_head.embed.weight` | BF16 | [129280, 256] | row gather by `output_ids[i]` |
| `mtp.2.markov_head.head.weight` | BF16 | [129280, 256] | GEMV [V ← 256] per draft row |
| `mtp.2.confidence_head.proj.weight` | BF16 | [1, 5376] | dot with `[head_in | markov_embed]` |

`embed.weight` and `head.weight` are the backbone's own (1212-1213) — already bound. The MTP
blocks' 30 backbone-shaped tensors are already bound under `mtp.N.` (`deepseek41.cpp:237-253`) with
`kind` = window-only, no engram, `n_routed/n_activated` = 128/3 (`deepseek41.hpp:54-62`).

### 6.2 The drafter forward: what exists in the engine vs what is new

| drafter op (reference line) | exists in the engine | new for P1 |
|---|---|---|
| `main_proj` FP8 [5120 ← 15360] (1130) | the FP8 dense path: `dense_proj` / `gemv_fp8_e4m3_f16` at T = 1 (`deepseek41_forward.cpp:665`), `ds41_dense_dequant_f16` (456) | the `[T, 15360]` input buffer from the capture; an `upload` role for one more FP8 matrix |
| `main_norm` (1130) | `ds4_rms_norm` (`include/ie/deepseek4_attn.hpp:133`) | — |
| noise-token rows (1131-1134) | the host-side embedding gather + hc broadcast (`forward.cpp:400-405`) | the same loop over the id list `[t_{p+1}, 128799 ×4]` |
| `hc_mixes` / `hc_pre` / `hc_post` (981-994) | `ds41_hc_mixes` (687, 832), `ds41_hc_collapse` (1058), `ds4_hc_mix` (816, 1040) | run at Tl = 5 at "decode": check the decode-shape scratch `hcs` (560, `decode ? … : nullptr`) is the T-row shape, since Tl ≠ 1 |
| q_a / q_norm / q_b / RoPE (1057-1059) | 694-698, `ds4_rope_apply` (`deepseek4_attn.hpp:112`) | RoPE tables at positions `p+1 .. p+5` — `rope_tables(rmain, pos_list, …)` (483) with base θ, no YaRN (686-687: `rmain` at 211-214 already is that) |
| the drafts' own kv / kv_norm / RoPE (1060-1061) | 699-701 | 5 rows, positions `p+1 .. p+5`, **never written to the ring** |
| `main_kv` and the ring write (1040-1041, 1065) | the ring write kernel (702-709) writes the layer's own `kvwn`; `LayerState.win_kv` (`deepseek41_forward.hpp:205-206`) | **new**: one row from `main_x` (wkv → kv_norm → RoPE at `p`) into slot `p % 128`; three more `LayerState`s (`card.state.assign(c.n_layers, …)` at `forward.cpp:192` → 43) |
| `cat([ring, kv])` + `sparse_attn` with the 128+5 index set (1064-1067) | `ds4_attention_segs` (`deepseek4_attn.hpp:189`; used at 803-804 with `segs.a` = ring, `segs.b` = compressed) takes exactly two KV segments and a mask | **new**: the mask builder — `[5, 133]`: ring column `k` open iff `k ≤ p` (the decode mask at 806-808, extended to 5 query rows), all 5 draft columns open (**no causal mask among the drafts**) |
| inverse RoPE on `o`, `wo_a` bmm, `wo_b` (1068-1074) | 810-814 | — |
| MoE 128 routed / top-3 + shared (`get_moe_config`, 142-149) | the router (`h_idx` path, 837-890), `Ds41ExpertTier::moe` (987), slot layout (`deepseek41_experts.cpp:265`) — the MTP expert bytes are the backbone's slot format | **new**: a second tier instance over layers 40-42 (`init` hardcodes `E_/TK_ = 384/6` at `experts.cpp:260` and rejects `first_layer + n_layers > c.n_layers` at 261); `TK` per layer from `k.n_activated` (the forward sizes `h_idx` with one `TK` at 412) |
| collapse + `mtp.2.norm` + tied head over 5 rows (1144-1145) | `ds41_hc_collapse` + `ds4_rms_norm` + the head GEMM with `TL` rows (1058-1067) | the norm weight is `mtp.2.norm.weight`; `TL = 5`, `logits_last_only` off for the drafter |
| Markov head (1077-1086, 1149-1153) | — | **new**: 5 sequential steps of gather [256] → GEMV [V ← 256] → add to row i → argmax → next id. 165 MFLOP total; a host loop over the `[5, V]` logits copy (as `sample` is host-side in `deepseek41_generate.cpp:30-53`) is adequate for P1 |
| confidence head (1089-1097, 1155) | — | **new**: 5 dots of length 5376; host-side |
| prefill seeding (1044-1052; docs/48 C.3) | the bounded-replay decoder half runs the last 128 prompt rows (`forward.hpp:183-188`) | **new**: `main_proj`/`main_norm` over the captured `[T, 15360]`, then a T-row ring write from `main_x` (slot = pos % 128); assert the segment-to-slot alignment |
| `main_hidden` capture (1264-1271) | — | **new**: at the top of iterations 37, 38, 39, mean `h [Tl, HC, H]` over HC into `main_hid[:, (L-37)·H …]`; at prefill too. The engine's `h` at the top of iteration L is exactly the golden's `layer_out_{L-1}` |

### 6.3 The P1 engine test: files, bars, reasons

The test forces `sp{pos}_main_hidden.f32` and `input_id` (from `sp_meta.json`) at `pos = 12, 13,
14, 15` in order after prefilling the 12-token prompt, mirroring `ds41_decode_test.cpp`'s
structure (`rd<T>` readers, `rel()` = max|a−b| / max|b| at :42-44):

| golden file | engine object | bar | reason |
|---|---|---|---|
| `sp_prefill_window_{s}.f32` (slots 0..11) | the 3 rings after prefill seeding, from the FORCED `sp_prefill_main_hidden.f32` | **3e-3** | one FP8 GEMV + RMSNorm + RoPE per slot — the same depth as layer 0's ring, which passes Phase 9 criterion 1 at 3e-3 (`decode_test.cpp:145`) |
| `sp{pos}_main_x.f32` | `main_x` | **3e-3** | one FP8 GEMV [5120 ← 15360] + RMSNorm |
| `sp{pos}_window_{s}.f32` (slots 0..pos) | the rings after the pass | **3e-3** | as above; also asserts the slot algebra |
| `sp{pos}_topk_{s}.i32` | the columns the mask opens | **identical** | discrete: 13+5 at pos 12 |
| `sp{pos}_embed.f32` | the 5 draft rows | **bit-identical** | bf16 rows → fp32, no arithmetic |
| `sp{pos}_attn_in_{s}`, `attn_out_{s}`, `ffn_in_{s}`, `moe_out_{s}` [1,5,5120]; `layer_out_{s}` [1,5,4,5120]; `ffn_pre_{s}` [1,5,4] — routing FORCED from `sp{pos}_route_idx_{s}.i32` / `route_w_{s}.f32` | the stage boundaries | **1.2e-2** per stage | the backbone's forced-routing bar for card 0 (`bar_of(L < 20)`, `decode_test.cpp:191, 210`): the drafter is 3 blocks deep from a forced input — the same depth as layers 0-2 — on the same FP8 dense + FP4 expert paths whose measured floor that bar is. Not 3e-3, because a block includes the Q8 grouped expert path and the Sinkhorn; not 3.5e-2, because that is card 1's 20-layer accumulation |
| `sp{pos}_head_in.f32` | `hc_pre(layer_out_2, ffn_pre_2)` | **1.2e-2** | a mix of `layer_out_2` |
| `sp{pos}_logits_base.f32`, `sp{pos}_logits.f32` [5, V] | the head rows, before / after the Markov bias | **5e-2** | the forced-logits bar (`decode_test.cpp:214`): a bf16 head GEMM over a 1.2e-2 input |
| `sp{pos}_markov_embed.f32` [5, 256] | the gathers | **bit-identical** | no arithmetic |
| `sp{pos}_markov_bias.f32` [5, V] | the 5 GEMVs | **1e-3** | a 256-long bf16 reduction on exact inputs |
| `sp{pos}_output_ids.i32` [6] | the argmax chain | **identical** — a mismatch is a hard failure UNLESS that row's `draft_margin` (in `sp_meta.json`) is below the logits bar × max|logits| (then it is a near-tie, reported not failed; the margin rule of `decode_test.cpp:273`) | discrete; row 0 is the input by construction |
| `sp{pos}_confidence.f32` [5] | the 5 dots | **1e-2** relative (report 1e-3) | docs/48 P1 criterion 2 says 1e-3; the confidence is a linear functional of `head_in`, which carries up to 1.2e-2 — 1e-3 is not guaranteed by the bars upstream, so 1e-2 is the defensible criterion and 1e-3 the number to report. The engine's dot must be fp32 as the reference's is (1092-1097) |
| `sp{pos}_route_idx_{s}` / `route_gap_{s}` | the engine's OWN router, second pass | flips counted per stage; a flip at `gap ≥ 3e-3` is a defect | Phase 7's rule (`decode_test.cpp:239`) |

Negative controls the test should carry, as the backbone tests do (`decode_test.cpp:319-320`):
(i) the draft rows' RoPE position off by one → the `attn_out_{s}` bar must FAIL; (ii) a causal
mask among the 5 draft columns → `attn_out_0` rows 0-3 must FAIL (the golden's `control_rel` at
position 12 is the magnitude to expect).

### 6.4 Prerequisite reads before building

docs/48 §C.3 (the residency/state/placement facts — the whole drafter is card-1-local), §A.3 facts
1-2 (the ring holds the BACKBONE's `main_kv`, never the drafts'; the block is bidirectional), and
§E/P1 criterion 6 (device bytes: 7.933 GB total, 7.22 GB of it the 384 MTP expert slots).

---

## 7. Open items

* The real golden run (§4) — a founder-scheduled foreground job once the GPU gate is idle (it is
  CPU-only, but 16 threads for ~2 min would perturb a timing gate).
* The wrapped-ring golden (§3.1): choose between the real long-prompt regen (cost Unknown) and the
  synthetic-`main_hidden` variant (~2 min, algebra only).
* `golden_regen.sh` should gain the `golden_dspark` line (outside this task's write set).
* The P1 engine test file (`tools/ds41_dspark_test.cpp`, by analogy) is not written; §6.3 is its
  specification.
