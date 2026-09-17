# V4.1 port — DSpark design brief: the drafter, the protocol, what the engine lacks, the cost model, the phase plan

**Status: RESEARCH ONLY. Nothing was built, nothing was run, no GPU was touched.** Written
read-only against the reference (`~/models/DeepSeek-V4.1-Flash/inference/model.py`), the
checkpoint's safetensors headers, the tech report, and the engine on branch
`deepseek4-vision-exp`. Every claim below is labelled **Verified** (read in the code/checkpoint),
**Derived** (arithmetic from a measured number, the arithmetic shown), or **Assumption /
Unknown** (not established by anything in the tree).

**Why this document exists:** the founder's target is 20 tok/s single-stream decode. The engine is
at **88.1-88.3 ms/token = 11.3 tok/s** at a 2,048-token context (docs/47, one build, two samples).
docs/30, 39, 44 and 47 all say no single-stream path reaches 20, and docs/47 names DSpark as "the
structural lever". This brief says exactly what DSpark is, exactly what the engine would have to
grow, and — the part that matters — **what the arithmetic says the answer is.**

> **The headline, up front.** DSpark amortises the dense body beautifully (31.6 ms/token → ~7
> ms/token at six rows) but the dense body is only 36% of the step. The other 64% is the routed
> experts' **link fetch**, and that cost scales with the **union** of the verified tokens' routed
> sets, which under independence grows *superlinearly per token* (misses/layer/token 1.79 at T = 1
> → 2.81 at T = 6, **+57%**). At the central link estimate, **full** acceptance of all five drafts
> gives **18.1 tok/s** — still short of 20. The outcome swings from 13.8 to 29.0 tok/s on two
> quantities nobody has measured: (i) how correlated adjacent tokens' expert routing is, and
> (ii) whether the two cards' links even out at T > 1 the way small-number variance says they
> should. **Both are measurable today, read-only, on the committed binary, with no new kernel.**
> That measurement is P0 of this campaign and it should gate the build. Separately, two
> *adjacent* levers worth ~20-31 ms/token at T = 1 alone (an even link split; the EP helper's
> serialisation) are prerequisites for DSpark's own arithmetic, because DSpark's cost is 80% link.
> Pineapples are spiky: the 20 tok/s claim is not established either way by this document.

---

## A. The drafter's forward, exactly

All line numbers are `inference/model.py` unless stated. **Verified** by reading.

### A.0 The shapes, from `config.json` `text_config` and `inference/config.json`

| field | value |
|---|---|
| `num_nextn_predict_layers` / `n_mtp_layers` | 3 (backbone layers 40, 41, 42) |
| `dspark_block_size` | **5** — five draft rows, five draft tokens per pass |
| `dspark_noise_token_id` | **128799** = `<｜System｜>` in `tokenizer.json` (**Verified**: a real trained special token reused as the draft mask, *not* an unused slot — a golden must use exactly this id) |
| `dspark_target_layer_ids` | **[37, 38, 39]** |
| `dspark_markov_rank` | 256 |
| `dspark_n_routed_experts` | **128** (the backbone's 384) |
| `dspark_num_experts_per_tok` | **3** (the backbone's 6) |
| `compress_ratios[40..42]` | **0, 0, 0** — sliding-window only, no compressed KV, no indexer |
| `window_size` | 128, shared with the backbone |

`ModelArgs.get_moe_config` (142-149) returns the DSpark expert counts for any `layer_id >=
n_layers`. The report's §2.4.3 (tr.txt:664-666) agrees: "three Transformer blocks with a sliding
attention window of 128 tokens. A single forward pass through these blocks computes base logits
for five draft positions in parallel."

### A.1 What the MTP layer reads from the backbone

`Transformer.forward` (1242-1272) collects, **at the top of each target layer's iteration and
before that layer runs**:

```
1264|            # the MTP head reads the attention input of its target layers, not their output
1265|            if i in self.target_layer_ids:
1266|                main_hiddens.append(h.mean(dim=2))
1267|            h, pre_mix = layer(h, start_pos, pre_mix, image_mask)
1271|        main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None
```

So `main_hidden` is `[B, T, 3*dim]` = the **mean over the 4 hyper-connection copies** of the
residual stream as it stands *entering* layers 37, 38 and 39, concatenated. **Not** after the
engram (layers 37-39 have none), **not** after `hc_mixes`, **not** after `hc_pre`. In engine terms
it is exactly the loop's `h` buffer at the start of iteration `L ∈ {37, 38, 39}`, reduced over the
`HC` axis.

### A.2 `forward_embed` — how the token and the hidden state enter (1128-1135)

```
1130|        main_x = self.main_norm(self.main_proj(main_hidden))
1131|        draft_input_ids = input_ids.new_full([input_ids.size(0), self.block_size], self.noise_token_id)
1132|        draft_input_ids[:, 0] = input_ids
1133|        x = self.embed(draft_input_ids)
1134|        x = x.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
1135|        return x, main_x
```

* `main_proj`: `Linear(dim * 3 -> dim)`. Checkpoint: `mtp.0.main_proj.weight` **F8_E4M3 [5120,
  15360]** with `mtp.0.main_proj.scale` **F8_E8M0 [160, 480]** → 32×32 blocks, i.e. exactly the
  shape `gemv_fp8_e4m3_f16` and `ds41_dense_dequant_f16` already consume. **Verified** from the
  shard header.
* `main_norm`: `mtp.0.main_norm.weight` BF16 [5120]. Both exist **only on stage 0** — one
  `main_proj`/`main_norm` in the whole checkpoint, shared by all three stages (1111-1114).
* The draft rows: row 0 embeds the **real** token the backbone just produced; rows 1-4 embed the
  noise token 128799. Then the embedding is broadcast to the 4 hc copies, and
  `make_identity_pre_mix` (1159-1163) gives a one-hot mix on copy 0 — the same entry the backbone's
  layer 0 uses.

### A.3 `DSparkAttention.forward` (1032-1074) — the attention kind and its cache

```
1034|        assert self.compress_ratio == 0
1039|        main_freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
1040|        main_kv = self.kv_norm(self.wkv(main_x))
1041|        apply_rotary_emb(main_kv[..., -rd:], main_freqs_cis)
1044|        if start_pos == 0:                       # PREFILL: seed the ring and return x untouched
...
1052|            return x
1055|        freqs_cis = self.freqs_cis[start_pos + seqlen : start_pos + seqlen + block_size]
1057|        qr = self.q_norm(self.wq_a(x)); q = self.wq_b(qr)...; apply_rotary_emb(q[..., -rd:], freqs_cis)
1060|        kv = self.kv_norm(self.wkv(x)); apply_rotary_emb(kv[..., -rd:], freqs_cis)
1064|        topk_idxs = get_dspark_topk_idxs(win, bsz, block_size, start_pos)
1065|        self.window_kv_cache[:bsz, start_pos % win] = main_kv.squeeze(1)
1066|        kv = torch.cat([self.window_kv_cache[:bsz], kv], dim=1)
1067|        o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
1068|        apply_rotary_emb(o[..., -rd:], freqs_cis, True)
1070-73|      o -> wo_a (block-diagonal over o_groups, einsum) -> wo_b
```

Five facts that decide the port:

1. **The ring holds the BACKBONE's representation, not the drafts'.** Each MTP layer's
   `window_kv_cache[pos % 128]` is written from `main_kv` — `wkv @ main_x`, one row per *accepted*
   position. The five draft rows' own `kv` is concatenated for this pass only and **never stored**.
   Consequence: **the drafter's KV state is a function of accepted positions alone, so a rejected
   draft leaves no trace in it.** That removes the MTP layers entirely from the rollback problem.
2. **The block is NOT causal within itself.** `get_dspark_topk_idxs` (1021-1029):
   ```
   1023|    matrix = torch.cat([torch.arange(min(window_size, start_pos + 1)),
   1026|                        window_size + torch.arange(block_size)])
   1029|    return matrix.int().view(1, 1, -1).expand(bsz, block_size, -1).contiguous()
   ```
   Every one of the five query rows gets the **same** index list: all filled ring slots plus **all
   five** draft KV rows. The drafter is bidirectional over its block. Consequence: **a 5-row draft
   pass cannot be reproduced by five 1-row passes** — P1's golden must be the 5-row pass.
   (Because order does not matter to `sparse_attn` and slot `p` holds position `p` while `p <
   128`, `arange(min(win, pos+1))` names exactly the written slots; once `pos+1 >= 128` it names
   all of them.)
3. **Positions.** `main_x` rotates at `start_pos` (one row); the draft rows rotate at `start_pos +
   1 … start_pos + 5`. With `compress_ratio == 0`, `Attention.__init__` (685-687) sets
   `original_seq_len, rope_theta = 0, args.rope_theta` → **no YaRN, base θ = 10000** — the engine's
   existing `rmain` table, just extended to five positions.
4. `softmax_scale = head_dim**-0.5` and a per-head `attn_sink` — identical to the backbone.
5. The MTP attention is **full size**: `mtp.N.attn.wq_b.weight` is [32768, 1280] (64 heads × 512),
   `wo_a` [8192, 4096], `wo_b` [5120, 8192]. A draft layer costs as much dense traffic as a
   backbone layer.

### A.4 `DSparkBlock.forward` (1122-1126) and the MoE

```
1122|    def forward(self, x, start_pos, pre_mix, main_x):
1123|        if start_pos == 0:
1124|            self.attn(x, start_pos, main_x)   # prefill only seeds the window KV cache
1125|            return x, pre_mix
1126|        return super().forward(x, start_pos, pre_mix, None, main_x)   # drafts are text: no VL bias
```

At decode this is the ordinary `Block.forward` (968-994): `hc_mixes` → `hc_pre` → `attn_norm` →
`DSparkAttention` → `hc_post`; then `hc_mixes` → `hc_pre` → `ffn_norm` → `MoE` → `hc_post`, and
the block returns `ffn_pre` for the next block's collapse. `image_mask = None`, so `gate_bias_vl`
is never selected for drafts.

The MoE is `MoE(layer_id, args)` with `get_moe_config` → **128 routed, 3 activated, 1 shared**.
Checkpoint (**Verified** from the shard headers): `mtp.N.ffn.experts.E.{w1,w3,w2}.weight` **I8**
with the halved last dim (w1/w3 [2304, 2560], w2 [5120, 1152]) and **F8_E8M0** scales (w1/w3 [2304,
160], w2 [5120, 72]) — **byte-for-byte the same FP4 slot format as a backbone expert**, 128 per
layer × 3 layers = 384 (layer, expert) pairs. The shared expert is FP8 [2304, 5120] ×2 and [5120,
2304], again the backbone's shape.

### A.5 `forward_head` (1137-1156) — the tied head, the Markov head, the confidence head

```
1144|        x = self.hc_pre(x, pre_mix)
1145|        logits = self.head(self.norm(x), full_logits=True)       # [B, 5, V], head TIED to the backbone's
1146|        output_ids = input_ids.new_empty(input_ids.size(0), self.block_size + 1)
1147|        output_ids[:, 0] = input_ids
1149|        for i in range(self.block_size):
1150|            logits_bias, markov_embed = self.markov_head(output_ids[:, i])
1151|            logits[:, i].add_(logits_bias)
1153|            output_ids[:, i + 1] = sample(logits[:, i], self.temperature)
1154|        markov_embed = torch.stack(markov_embeds, dim=1)
1155|        confidence = self.confidence_head(x, markov_embed)
1156|        return output_ids, logits, confidence
```

* **Head tying** (1206-1213): `self.mtp[-1].head = self.head` and `self.mtp[-1].embed = self.embed`
  — the drafter reuses the backbone's `head.weight` [129280, 5120] BF16 and `embed.weight`.
  `convert.py:108-109` says the same ("an MTP layer ties its token embedding and output head to
  the backbone's"). **No extra head weight to upload.**
* `mtp.2.norm.weight` BF16 [5120] is the drafter's own final norm (`stage_id == n_mtp_layers - 1`,
  1115-1116).
* **The Markov head** (1077-1086):
  ```
  1084|        embed = self.embed(token_ids)                 # [B, 256]  markov_head.embed  BF16 [129280, 256]
  1085|        logits = self.head(embed, full_logits=True)    # [B, V]    markov_head.head   BF16 [129280, 256]
  ```
  It is a **rank-256 bigram correction**: the token chosen at draft position *i* contributes an
  additive bias to position *i*'s logits. This is the only sequential dependency in the drafter
  — the transformer body ran once over all five rows in parallel; the Markov head supplies "what
  did I just decide" for the cost of a 256-dim embedding lookup and one [256 → 129280] matvec per
  position. The report (tr.txt:665-666): "a lightweight Markov head models dependencies among the
  draft tokens."
  **Note the indexing:** the bias applied to position *i* is derived from `output_ids[:, i]`, i.e.
  from position *i*'s **input** token (position 0's input is the backbone's token; position *i*'s
  input is the token sampled at position *i−1*).
* **The confidence head** (1089-1097):
  ```
  1093|        self.proj = Linear(input_dim, 1, dtype=torch.float32)     # input_dim = dim + markov_rank = 5376
  1096|        hidden = torch.cat([hidden, markov_embed], dim=-1)
  1097|        return self.proj(hidden.float()).squeeze(-1)
  ```
  `mtp.2.confidence_head.proj.weight` BF16 [1, 5376]; the comment at 1092 says the checkpoint is
  BF16 and the parameter is fp32 "for fp32 confidence score". Output: **one scalar per draft
  position**, `[B, 5]`. Input: the hc-collapsed pre-norm hidden `x` and the stacked
  `markov_embed`.

### A.6 `forward_spec` (1274-1282) and the position algebra

```
1275|    def forward_spec(self, input_ids, main_hidden, start_pos=0):
1276|        h, main_x = self.mtp[0].forward_embed(main_hidden, input_ids)
1277|        pre_mix = make_identity_pre_mix(h, self.hc_mult)
1278|        for layer in self.mtp:
1279|            h, pre_mix = layer(h, start_pos, pre_mix, main_x)
1280|        if start_pos == 0: return None
1282|        return self.mtp[-1].forward_head(h, pre_mix, input_ids)
```

`main_x` is computed **once** and handed to all three stages; each stage writes its **own** ring.
With `generate.py`'s convention (`model.forward(tokens[:, prev_pos:cur_pos], prev_pos)` at
generate.py:69-76 — a 1-row forward at `start_pos = p` consumes the token at position *p* and
predicts position *p+1*), and the smoke test's call order (1305-1309):

| object | position |
|---|---|
| `main_hidden` | *p* (the backbone's input row) |
| `input_ids` given to `forward_spec` | *p+1* (the backbone's just-sampled token) |
| draft row *j*, *j* = 0..4 | *p+1+j* |
| `output_ids` = `[t_{p+1}, d_1 … d_5]` | positions *p+1 … p+6* |

So **one DSpark pass yields five draft tokens for positions p+2 … p+6**, and the verify pass must
consume the six tokens at *p+1 … p+6*. Maximum six tokens committed per backbone step.

### A.7 Drafter weight bytes (**Verified** from the shard headers of `model-0004{4,5,6}-of-00048`)

| group | bytes |
|---|---|
| routed experts, 3 × 128 slots | **7.219 GB** (18.80 MB per slot: 3 nibble planes 5,898,240 B + 3 scale planes 368,640 B) |
| attention, 3 layers (`wq_a wq_b wkv wo_a wo_b` + norms + sink) | 0.500 GB |
| shared experts, 3 layers | 0.106 GB |
| hc tensors + gates + norms, 3 layers | 0.024 GB |
| `main_proj` + `main_norm` (stage 0) | 0.079 GB |
| `markov_head.embed` + `markov_head.head` (stage 2) | **0.132 GB** |
| `confidence_head.proj`, `norm` (stage 2) | ~0 |
| **total `mtp.*`** | **7.933 GB** |

The drafter's *dense* set is 0.71 GB. **Its expert pool is 7.22 GB — ten times its dense set, and
the single biggest residency decision in the campaign** (see D.4).

---

## B. The draft-verify protocol

### B.1 What the report specifies (**Verified**, tr.txt:657-676)

The whole of §2.4.3 is thirteen lines. In full, the operative part:

> "The drafter comprises three Transformer blocks with a sliding attention window of 128 tokens. A
> single forward pass through these blocks computes base logits for five draft positions in
> parallel, while a lightweight Markov head models dependencies among the draft tokens. A
> confidence head predicts per-position conditional acceptance probabilities, which are used to
> estimate prefix survival probabilities. The scheduler combines these estimates with profiled
> engine throughput curves to dynamically select the verification length for each request, aiming
> to maximize expected system-wide token throughput under the current system load."

Three things follow, and one thing does **not**:

* **The confidence head does NOT gate acceptance.** It predicts per-position conditional
  acceptance probabilities; those are multiplied into **prefix survival probabilities**; and those
  feed a **scheduler that chooses how many of the five drafts to send to verification**. The
  accept/reject decision itself is the verifier's. This is a **Verified** reading of the report and
  it differs from the obvious guess.
* **The verification length is dynamic**, chosen per request against "profiled engine throughput
  curves" — i.e. the report itself expects the optimum to be hardware-dependent. Section D makes
  this engine's curve.
* **No numbers.** `grep -i "accept|speedup|tok/s|verification length"` over the whole report
  returns only the two passages above and the reference entry (Cheng et al., 2026a, *"Dspark:
  Confidence-scheduled speculative decoding with semi-autoregressive generation"*, arXiv). **There
  is no reported acceptance length, no reported speedup, and no reported per-position acceptance
  rate anywhere in the tech report.** Nor is there a DSpark figure — Figure 3 (tr.txt:292-297)
  shows only the block in the architecture diagram. **Unknown**, and the DSpark paper itself is not
  on disk.

### B.2 What the reference code implements — and what it does not (**Verified**)

`ModelArgs`' own comment, 129-130:

```
 129|    # dspark draft head. Only the forward pass is implemented here -- nothing calls forward_spec,
 130|    # so these are read but the speculative-decoding loop itself is out of scope for this repo.
```

Confirmed by grep: `generate.py` contains **no** occurrence of `spec`, `mtp`, `dspark`, `draft` or
`accept` (its only hit is the substring in "specified" at line 209); its loop (generate.py:67-80)
calls `model.forward(...)[0]` one token at a time and **discards `main_hidden`**. `convert.py`
touches `mtp.*` only for weight conversion (68-75, 96-98, 108-109, 130). The only caller of
`forward_spec` anywhere is the `__main__` smoke test (1295-1309), which drafts but never verifies.

**So the draft/verify protocol is not in the reference and not in the report. It must be
reconstructed.** What follows is the standard MTP/EAGLE-style protocol, stated as an
**Assumption**, with the pieces the reference *does* pin down marked.

### B.3 The protocol (Assumption, except where marked)

State before a pass: the backbone's caches hold positions `0 … p` (`n_pos_ = p+1`); the drafter's
three rings hold the same positions; the last backbone forward produced `main_hidden` at position
*p* and a token `t_{p+1}`.

1. **Draft** (**Verified** — this is `forward_spec` exactly).
   `forward_spec(input_ids = t_{p+1}, main_hidden, start_pos = p)` → `output_ids = [t_{p+1}, d_1 …
   d_5]`, `logits [5, V]`, `confidence [5]`. Cost: three MTP layers at 5 rows, plus five sequential
   Markov steps. Side effect: each MTP layer writes `main_kv(p)` into ring slot `p % 128`. **This
   side effect is unconditional and correct regardless of what is later accepted** (A.3 fact 1).
2. **Choose the verification length** *k* ≤ 5 from `confidence` (**Verified** that this is the
   report's design; the *formula* is **Unknown**). The natural form, given "prefix survival
   probabilities": with `c_i = σ(confidence_i)` treated as conditional acceptance probabilities,
   `S_k = Π_{i≤k} c_i` is the probability the *k*-prefix survives, `E[tokens | k] = 1 + Σ_{j≤k}
   S_j`, and *k* maximises `E[tokens | k] / C(1 + k)` against the engine's own cost curve `C`.
   **Section D supplies `C` for this hardware; the calibration of `c_i` from the raw fp32 scalar is
   Unknown and must be fitted.** For the first build, fix `k = 5`.
3. **Verify.** One backbone `forward(ids = [t_{p+1}, d_1 … d_k], T = 1 + k, pos0 = p + 1)` with
   **all `T` rows' logits returned**. Row *r* (0-based) sits at position `p+1+r` and predicts the
   token at position `p+2+r`.
4. **Accept, greedily** (**Assumption**; exact and lossless at temperature 0). Let `a_r =
   argmax(logits[r])`. Accept `d_{r+1}` iff `d_{r+1} == a_r` and every earlier draft was accepted.
   Let `L` = 1 + the length of that matching prefix — so `1 ≤ L ≤ 1 + k`. The committed tokens are
   `t_{p+1}` plus the accepted drafts, **and then one bonus token**: `a_{L-1}`, the verifier's own
   argmax at the last accepted row, which is by construction what the backbone would have produced
   next. (This is the standard free extra token; it makes `L = 1 + k` yield `k + 1` *new* tokens
   from one verify pass plus the previous pass's token.)
   At temperature > 0 greedy comparison is **not** lossless; losslessness there needs the standard
   rejection-sampling correction against the drafter's own `logits` (available, but the reference's
   `sample` is Gumbel-max at 1285-1292 while the engine's is a host-side CDF walk at
   deepseek41_generate.cpp:30-53 — two different algorithms). **Recommendation: scope the lossless
   criterion to temperature 0 and treat speculative sampling as a later phase with its own
   criteria.**
5. **Roll back to `L`.** Everything the verify pass wrote for rows `L … k` must be undone, and
   `n_pos_` must become `p + 1 + L`. Section C.2 shows this costs a few hundred KB of snapshots and
   two counters — no recompute.
6. **Re-draft** from the new last position, using the verify pass's `main_hidden` at row `L-1`.

**Steady-state accounting.** One pass = one drafter forward + one verify forward, and commits `L`
tokens where `1 ≤ L ≤ 1 + k`. At `k = 5`: `L ∈ [1, 6]`.

---

## C. What the engine lacks

### C.1 A decode forward at T > 1 at pos0 > 0

Today refused outright:

```
src/model/deepseek41_forward.cpp
 317|    const bool decode = pos0 > 0;
 324|        if (T != 1) return "forward: at pos0 > 0 only single-token steps are admitted (T == 1)";
 325|        if (pos0 != n_pos_) return "forward: step at pos0 ... but the state holds ... positions";
```

Every place in `deepseek41_forward.cpp` that assumes `T == 1` at decode, **Verified** by reading,
with what a `T > 1` step needs. ✅ = already T-generic, no change.

| # | site | lines | today | needed at T > 1 |
|---|---|---|---|---|
| 1 | admission | 324-325 | refuses `T != 1` | admit `pos0 == n_pos_` with `1 ≤ T ≤ 1+block_size` |
| 2 | engram hashes, host | 359-367 | the decode branch hashes the last `min(n, max_ngram)` ids and keeps **one** row (`hash.assign(hw.end() - ENC, hw.end())`) | hash the last `min(n, max_ngram - 1 + T)` ids, keep the last `T` rows. The reasoning at 352-354 already justifies the window; only the row count changes. The consumer at 612 already indexes `T` rows ✅ |
| 3 | RoPE tables, fast path | 477-491 | five **scalar** position entries: `[0]=pos_t[0]`, `[1]=pos_qk[0]`, `[2]=pos_qk[0]`, `[3]=pos0-1` (the ratio-2 latent), `[4]=pos0` (the ratio-1 latent) | `T` q/kv positions + one position per **emitted latent** per ratio. The `rope_pos`/`rope_cs`/`rope_sn` buffers are sized `[8]` / `[8, RD/2]` — need `2T + latents` |
| 4 | RoPE tables, slow path | 499-503 | `rope_tables(rcomp, {pos0-1})`, `{pos0}` — single-element vectors at decode | the same per-latent position lists |
| 5 | window ring write | 682-689 | `nwr = min(Tl, WIN)`, slot `(pos0 + seg_off + first + r) % WIN` | ✅ **already correct for any T** |
| 6 | compressor, ratio 1 | 715-717 | `n_new = 1`, one `rms` row | `n_new = T`; rms over `T` rows; `T` RoPE positions `pos0 … pos0+T-1` |
| 7 | compressor, ratio 2 | 718-733 | `(pos0+1) % R != 0` → hold the row+gate in `part_*`; else pool `[part; new]` through `ds4_compress_pool(..., 1, R, HD, false)` | the step spans `T` positions and closes `⌊(pos0+T)/2⌋ − ⌊pos0/2⌋` groups. Assemble a contiguous `[n_new*R, HD]` buffer from `part_*` (when the parity opens on a held half) plus this step's rows, call `ds4_compress_pool` **once** with the real `n_new`, and hold the trailing odd row. **The most intricate change in the list** |
| 8 | index-key append | 734-745 | `n_new` rows through `idx_wk`+rms+RoPE, but with a **single-row** `cos_g/sin_g` table (741, 743) | a `[n_new]` position table. **Latent bug**: at `n_new > 1` the current decode call would rotate every latent at the *same* position. The prefill path passes a `T/R`-row table (497-498); the decode path a 1-row one (501-502) |
| 9 | indexer score | 755 | `ds4_indexer_score(..., Tl, ...)` | ✅ T-generic |
| 10 | indexer top-k + causal threshold | 763 | `ds4_indexer_topk(q, scores, d_pos_t, sh_topk, Tl, NC, index_topk, R)`; the kernel computes `threshold[t] = (positions[t]+1)/rate` per row (deepseek4_attn.hpp:347-364) | ✅ **T-generic and already correct**, including the subtlety that row *r* must not see latents this same step produced for rows > *r* — the position-based threshold handles it exactly. **But** `d_pos_t` under `dec_cache` is `card.rope_pos` (487), an 8-entry scalar buffer; it must become a `T`-entry table of *real* positions |
| 11 | `sh_topk_w` | 762 | `min(index_topk, NC)`, one scalar shared by all rows | ✅ correct (the kernel emits that many per row) |
| 12 | decode mask, ratio > 0 | 774-784 | a **one-row `[NKV]`** mask: `mask[kk] = kk < WIN ? (kk <= pos ? 0 : LOWEST) : mcomp[kk - WIN]` | `[Tl, NKV]`, with (a) `mcomp[t*NC + (kk-WIN)]`, and (b) **per-row ring causality**. `kk <= pos` is a ring-*filling* test, valid only while `pos < WIN`; at `T > 1` the ring holds this step's own rows, so row *r* must exclude the `T-1-r` slots holding positions > `pos0+r`. The general test: with `p_last = pos0+T-1`, slot `kk` holds position `q = kk + WIN·⌊(p_last − kk)/WIN⌋`; open for row *r* iff `0 ≤ q ≤ pos0+r`. Two lines |
| 13 | decode mask, ratio 0 | 787-790 | `mask[i] = i <= pos ? 0 : LOWEST`, `[WIN]` only | the same `[Tl, WIN]` generalisation |
| 14 | `ds4_attention_segs` | 773, 783 | called with `Tl` rows and a `[Tl, n_a+n_b]` mask — the prefill path at 768-773 already does exactly this | ✅ **T-generic.** (`ds4_attention_xmx_eligible` needs `T > 16`, so a 6-row step takes the fp32 kernel, as decode already does) |
| 15 | `ds4_block_bias_topk` | 765 | `(q, sh_topk, mcomp, Tl, NC, sh_topk_w)` | ✅ T-generic |
| 16 | **`hcs` / `ds41_hc_mixes`** | 539, 666, 811 | `hcs` is allocated at decode and passed; but `ds41_hc_mixes` takes the chunked decode shape **only at `n_tokens == 1`** (src/ops/deepseek41_ops.cpp:77). At `T = 6` it silently falls to the general kernel (138-194), which launches `n_tokens` work-groups and gives each of the 24 mix rows to **one lane** walking all 20,480 elements serially — **the ~700 µs/call shape Phase 11 measured**. 2 calls × 40 layers ⇒ **~56-72 ms/step**. **Derived**, and a hard blocker | generalise the chunked shape to `T` rows: `(mix+1) × kDs41HcChunks × T` work-groups, each row chunked identically. **This is also what makes P2's bit-identity criterion reachable** — the T-row chunked shape is bit-identical to the 1-row chunked shape per row, whereas the general kernel is not |
| 17 | head rows | 1031-1036 | `TL = logits_last_only_ ? 1 : Tl` — the generator sets `logits_last_only(true)` (deepseek41_generate.cpp:62) and `Engine::ds41_load` sets it too (src/engine/ds41_engine.cpp:52) | run the verify with `logits_last_only(false)` → `[6, V]` fp32 = 3.1 MB D2H, trivial. No kernel change. The `bounded_replay` interlock at 340 is prefill-only (`replay = bounded_replay_ && !decode && T > WIN`) so decode never trips it ✅ |
| 18 | persistent decode scratch | 406-423 | `take()` matches on **exact byte size** (417) and returns `nullptr` on a mismatch, falling through to a fresh `malloc_device` that is *not* cached. A `T = 6` step therefore pays back the ~48 allocations per card per step that Phase 18's term 2 removed (**~3 ms**, docs/47) | key `dec_scratch` on `T` (one list per admitted `T`) |
| 19 | `all_ids_` / `n_pos_` | 346-348, 1066 | the `RollBack` guard resizes `all_ids_` on error; `n_pos_ = pos0 + T` on success | correct for a full commit; a partial commit needs `resize(pos0 + L)` and `n_pos_ = pos0 + L`. Host-side, trivial |
| 20 | the CPU miss split | experts.hpp:203-207 | `T = 1` only; and docs/47 records its EP-export path as broken (a null `y` in export mode) | off at `T > 1` today. **But see D.4 — at T = 6 it is the most attractive lever in the engine** |
| 21 | the source-cache bounce elision | 1046-1051 | `dead_bounce = dec_cache && nk.is_kv_source && nk.is_index_source` | ✅ still valid at `T > 1` (layer 20 re-derives both caches before any consumer reads them) |
| 22 | tier workspace | experts.cpp:464 | `if (T > bws_.max_tokens)`, `max_tokens = 2048` | ✅ fine |

Scratch sizes at decode are `T × cap_pos_` for `scores`, `mcomp`, `mask` (524-535) — at `T = 6`
that is 49 KB each. No pressure.

### C.2 Rollback after rejection — and why it is cheap

**The key structural fact (Derived, from the causal mask):** because the verify pass's attention is
causal across its rows, **row *r*'s hidden state, ring KV row, compressor input and index key at a
`T`-row step are bit-identical to the same row's at an `L`-row step, for every `r < L`.** So the
accepted prefix's state is *already* exactly right; only the rejected rows' writes must be undone.
There is **no recompute**.

Per layer, per card, what the verify pass mutates and how to restore it:

| state | mutation | restore |
|---|---|---|
| `win_kv` (ring, `[128, 512]`) | rows 0…T-1 written at slots `(pos0+r) % 128` (685-689). The rejected rows' slots **alias slots the next window still needs** (for `T > L`, positions `pos0+L+1-128 … pos0+T-1-128` are all ≥ `pos0+L-127`), so they **must** be restored | snapshot the ≤ `T-1` target slots before the ring write: `(T-1) × 512 × 4` = **10 KB per layer**, 400 KB for 40 layers. Restore slots `L…T-1` on rejection |
| `comp_kv`, `idx_k` | appended past `nc` (737-744); nothing past `nc` is ever read (`NC = cur.nc`, 749) | **restore the counter alone.** `nc_accepted = (pos0 + L) / R`. Free |
| `part_kv`, `part_gate`, `part_valid` (ratio-2 sources: layers 2, 8, 14) | opened/closed across the step | snapshot before (`2 × 512 × 4` = 4 KB × 3 layers = 12 KB) **and** save each row's `ckv_p`/`cg_p` for those 3 layers (`T × 512 × 2 × 4` = 24 KB each, 74 KB total). After acceptance `L`, the correct partial is empty (even count) or exactly row `L-1`'s pair — pick it from the saved rows. **Exact, no recompute** |
| the three MTP rings | written from `main_kv(p)` only, one accepted position per pass | **nothing to do** (A.3 fact 1) |
| `nc` on the imported caches (`imp_ckv`, `imp_ik`) | overwritten by the bounce | derived from the owner's `nc`, restored with it |
| `all_ids_`, `n_pos_` | 347, 1066 | `resize(pos0+L)`, `n_pos_ = pos0+L` |
| `last_hashes_` | informational | — |
| the expert stream-slot cache, the dense cache | pure caches, no correctness dependence | nothing |

**Total: under 500 KB of device snapshots and two counters per step.** Snapshot/restore beats
recompute by a very wide margin, and it is the design this brief recommends.

### C.3 The MTP layers' weights and state

**The weights are already bound.** `DeepSeek41Model::load` loops over `n_total = n_layers +
n_mtp_layers = 43` (deepseek41.cpp:237, 249) with the prefix `"mtp." + (L - n_layers) + "."` (253),
and `layers()`' own comment says `n_layers + n_mtp_layers` (deepseek41.hpp:152). `kind` comes out
right by construction: `compress_ratio = compress_ratios[40..42] = 0`, and `is_kv_source`,
`is_index_source`, `has_engram` are all gated on `backbone` (258-266) exactly as the reference
gates them on `is_backbone` (model.py:653-655). `n_routed`/`n_activated` come from `Ds41Config::
n_routed(L)` / `n_activated(L)` → **128 / 3** (deepseek41.hpp:54-62). **Verified**, and
independently confirmed against the checkpoint: `mtp.N`'s 30 non-expert tensor names are *exactly*
`layers.39`'s 30 names.

What is **not** bound — six tensors currently swallowed by the `mtp_extra` claim at
deepseek41.cpp:365-371, which exists only so `unclaimed()` stays empty:

```
mtp.0.main_proj.weight   F8_E4M3 [5120, 15360]   +  mtp.0.main_proj.scale  F8_E8M0 [160, 480]
mtp.0.main_norm.weight   BF16    [5120]
mtp.2.markov_head.embed.weight   BF16 [129280, 256]
mtp.2.markov_head.head.weight    BF16 [129280, 256]
mtp.2.confidence_head.proj.weight BF16 [1, 5376]
mtp.2.norm.weight        BF16    [5120]
```

So P1's binding work is: one `Ds41Dspark` struct, six `b.bind` calls, and deleting the
corresponding clauses from `mtp_extra`.

What else is missing:

* **Upload.** `Ds41DenseCache::upload_layer(q, m, L)` works **as is** for `L = 40, 41, 42`: it
  branches only on `kind`, which already says "no compressor, no indexer, no engram"
  (deepseek41_weights.cpp:76-90). Only the six DSpark roles need their own upload path.
* **Experts.** `Ds41ExpertTier::init` hardcodes `E_ = c.n_routed_experts` (= 384) and `TK_ =
  c.n_activated_experts` (= 6) at deepseek41_experts.cpp:260. The MTP layers are 128 / 3 → they
  need a **separate tier instance** over layers 40-42 with its own `E`/`TK` and its own ranking
  (cleanest), or per-layer `E`/`TK` in the tier.
* **State.** `ensure_state` does `card.state.assign(c.n_layers, ...)` (forward.cpp:177) — 40
  entries. Needs 43; the three extra window rings are 256 KB each.
* **Placement.** The card split is `[n_layers·c/n, n_layers·(c+1)/n)` and the head lives on the
  last card (forward.cpp:112-127). Since target layers 37-39 are on card 1 (which owns 20-39) and
  the head is there too, **the whole drafter is card-1-local — no cross-card bounce.**
* **`main_hidden` capture.** One kernel: at the top of iteration `L ∈ {37, 38, 39}`, mean-reduce
  `h` `[Tl, HC, H]` over `HC` into `main_hid[:, (L-37)*H … ]`, a `[Tl, 3H]` buffer (`Tl × 15360 ×
  4` = 368 KB at T = 6). Must run at **prefill** too (see below).
* **Prefill seeding.** At `start_pos == 0` the drafter runs its **attention only** (model.py:1123-
  1125) to seed the rings; `forward_spec` returns `None` (1280-1281). So after a prompt prefill the
  engine must run `main_proj`/`main_norm` over `main_hidden` for the prompt positions and push the
  last 128 rows' `wkv` output into the three rings. **Happy alignment (Verified):** with
  `bounded_replay` on (the default, forward.hpp:174-181) the decoder half runs over the last
  `window_size = 128` prompt tokens only — which is **exactly** the 128 rows the drafter's ring
  needs. Row *r* of the segment has position `pos0 + seg_off + r` and lands at slot `p % 128`, so a
  128-row segment fills the ring exactly. The alignment must be *checked*, not assumed.
* **Activation quantisation.** The reference calls `act_quant(main_kv, ...)` / `act_quant(kv, ...)`
  (1042, 1062) on the drafter's KV. The port's golden harness replaces `act_quant` and
  `fp4_act_quant` with identity (docs/09, table row 1; docs/26:33-34) and the engine keeps both
  caches in fp32 — the drafter gets the same treatment, **no new decision**.

### C.4 The sampling / acceptance loop and the server

`Ds41Generator::run` (deepseek41_generate.cpp:55-103) is strictly one token per `forward` call:

```
  66|    fwd_.forward(prompt_ids.data(), Te, 0, logits)        // prefill, even prefix
  67|    fwd_.forward(&prompt_ids[Te], 1, Te, logits)          // the odd tail as a 1-token step
  78|        const int32_t id = sample(logits, recent, sp, rng);
  97|        fwd_.forward(&id, 1, pos, logits)
```

`sample` (30-53) is a host-side single-row sampler over a `[V]` vector (repeat penalty → argmax at
`temperature <= 0`, else partial_sort / top-k / top-p / min-p / CDF walk).

Needed: a `Ds41SpecGenerator` (or a mode on the existing one) that (i) sets
`logits_last_only(false)` for the verify, (ii) runs the drafter, (iii) calls `forward` at `T = 1+k`,
(iv) slices the `[T, V]` block per row, (v) compares argmax with the drafts, (vi) calls the
rollback, (vii) drives the detokeniser/stop logic for `L` tokens at once (the existing UTF-8 and
stop-string machinery at 81-95 works unchanged if fed one token at a time in order).

The **server path needs no change**: `src/engine/ds41_engine.cpp` is 110 lines — `ds41_load` sets
`logits_last_only(true)` at :52 and `chat()` constructs a `Ds41Generator` at :74. Swapping the
generator class (and moving the `logits_last_only` decision inside it) is the whole adapter change.

---

## D. The cost model

### D.1 The measured baseline (all **Verified**, docs/46 and docs/47, one build, 2,048-token context, `IE_DS41_EP=1 IE_DS41_EXPERT_FILE IE_DS41_DENSE_FP8=1 IE_DS41_FP8_PACKED=3 IE_DS41_SHARED_EARLY=1 IE_DS41_SETUP_FAST=1`)

**Step: 88.1 / 88.3 ms/token = 11.35 / 11.33 tok/s.** Columns (docs/47, step 1; docs/46 for the
terms Phase 18 did not move): MoE 62.2 (prep 0.6, groups 38.7, tail 19.4), attention 25.0, ffn_pre
3.1, head 2.3, prep+engram 1.1, shared 0.4 (now hidden inside the MoE), setup 0.1. Named kernel
device time 45.8 ms/token: FP8 GEMV 14.6, expert GEMMs 12.1, attention 7.0, `o_a` 4.7, head 2.2,
router 1.5, the rest < 1 each. docs/46 states the attention site's 25 ms wall now matches its ~23 ms
of device time — **the dense body is kernel-bound, with no host gap left.**

**Exposed split (Derived):** dense body = attention 25.0 + ffn_pre 3.1 + head 2.3 + prep/engram 1.1
+ setup 0.1 = **31.6 ms** (37.0 before Phase 18 hid the shared expert's 5.4); MoE = 88.1 − 31.6 =
**56.5 ms**. So **64% of the step is the expert fetch.**

**Per token:** 1,262.9 MiB over the links + 22.4 MiB from disk; experts 116.3 static / 122.5 pinned
/ 1.2 mmap of 240 selected (40 layers × 6); 52.0 stream hits; hit rate 70.1%. EP: the slower card
serves 153.9 of 240 selections with **link misses 55.2 / 15.3**; the remote tiers' wall 36.1-36.8 ms.

**Two calibrations that anchor everything below (Derived):**

* **Bytes per miss.** 1,262.9 MiB / (122.5 − 52.0 + 1.2) = 1,262.9 / 71.7 = **17.91 MiB**.
  Independently computed from the checkpoint: 3 × (5,898,240 nibble + 368,640 scale) = 18,800,640 B
  = **17.93 MiB**. Agreement to 0.1% ✓.
* **The slower link is already saturated.** Its share is 55.2/70.5 = 78.3% of 1,262.9 MiB = 988.6
  MiB = 1.0366 GB, moved inside the 38.7 ms groups column ⇒ **26.8 GB/s** — at the 26-26.5 GB/s
  single-link ceiling (docs/39:94-95; the hardware note's 26.5 GB/s measured PCIe). **So the fetch
  is not merely link-bound, it is saturating one link. Every extra miss byte costs time 1:1.**

### D.2 The union of the verified tokens' experts

Per layer at `T` rows there are `6T` selections from 384 experts, and the tier fetches the **union**
(experts.cpp:499-500: `if (off[e+1] <= off[e]) continue` — an expert is fetched iff at least one row
picked it).

**A two-mass independence model**, calibrated so that `T = 1` reproduces the measurement exactly:
the 43 static experts per layer (22 + 21 across the cards) carry 2.908 of the 6 selections
(`p_static = 2.908/43 = 0.06763` each), the other 341 carry 3.092 (`p_tail = 3.092/341 =
0.009067`). `E[union ∩ S] = Σ_{e∈S}(1 − (1−p_e)^T)`.

| T | static∩union | tail∩union | union | misses/layer (tail∩union − 1.30 stream hits) | **misses/layer/token** |
|---|---|---|---|---|---|
| 1 | 2.91 | 3.09 | 6.00 ✓ | **1.79** ✓ | 1.79 |
| 2 | 5.62 | 6.15 | 11.77 | 4.85 | 2.43 |
| 3 | 8.15 | 9.19 | 17.34 | 7.89 | 2.63 |
| 4 | 10.51 | 12.20 | 22.71 | 10.90 | 2.72 |
| 5 | 12.71 | 15.18 | 27.89 | 13.88 | 2.78 |
| 6 | 14.76 | 18.13 | **32.90** | **16.83** | **2.81** |

`T = 1` reproduces 6.00 selections and 1.79 misses by construction, and the resulting `T = 1` link
bytes (40 × 1.79 × 17.93 MiB = 1,285 MiB) match the measured 1,263 MiB to 2% — a weak but real
check that the model is not nonsense.

**The bad news, stated plainly (Derived):** misses **per token** rise 57% from `T = 1` to `T = 6`.
The static tier saturates first (its experts are the popular ones, so they are the ones the six
rows duplicate), the 8 stream slots per layer cannot absorb 18 distinct fetches within one step, and
what is left is 18 tail experts that are almost all distinct. **The expert fetch does not amortise.
It anti-amortises.**

**Assumption, and the one that decides the campaign:** adjacent tokens' routing is **correlated**,
not independent. Published MoE studies report neighbouring-token expert overlap far above chance
(chance here is 6/384 = 1.6%). If the effective number of independent draws is `T_eff = 1 +
0.6(T−1)` (a 40% correlation discount — a *guess*, calibrated by nothing), then `T = 6` behaves
like `T_eff = 4` and misses/layer fall from 16.83 to 10.90, a **35% reduction in link bytes.**
**This is Unknown and it is measurable today** — see F.1.

### D.3 The per-pass cost, and where 20 tok/s lands

**Dense body at `T` rows (Derived).** The weight-bound terms are flat in `T` (the same weights read
once): FP8 GEMV 14.6 + `o_a` 4.7 + head 2.2 + router 1.5 + misc ≈ 2 = 25.0 ms. The growing terms:
attention (7.0 at `T = 1`, 175 µs/call against its own 80-90 µs bench, i.e. occupancy-starved, so
the marginal query row is cheap — estimate +1.6 ms per extra row) and the engram gather (1.1 ms at
`T = 1`, 24 random 6 KB rows per position into a 91.5 GB mmap per engram layer — estimate +0.7 ms
per row). **`dense(T) ≈ 31.6 + 2.3(T−1)` ms.** The attention term is the least certain.

**Two traps that must be fixed or the dense body explodes (Derived, both blockers):**

1. **The FP8 dense path at `T > 1` costs 5 bytes per weight, not 1.** `dense_proj` (642-648) takes
   the GEMV only at `T_ == 1`; above it, `card.cache.f16(q, *f8)` (weights.cpp:95-105) dequantises
   the **whole** matrix into one shared scratch per call ⇒ read 1 (FP8) + write 2 (fp16) + read 2
   (GEMM) = 5 bytes/weight against the GEMV's 1. The nine converted projections are ~5.5 GB of FP8
   per token ⇒ **~27.5 GB, ~50-70 ms.** Two fixes: (a) `IE_DS41_DENSE_FP8=0` → fp16 resident, 11 GB
   at the measured ~580 GB/s = **19.0 ms** flat in `T`, at the cost of 5.1 GiB more VRAM (14.45 vs
   9.33 GiB, docs/44) ≈ 4 fewer static slots per layer per card ≈ +110 MiB per **pass** (amortised
   6×, trivial); or (b) extend `gemv_fp8_e4m3_f16` to a small-`M` shape (`M ≤ 8`, `M` accumulators
   per output column, still 1 byte/weight) → **14.6 ms.** (b) is better and the kernel already has
   four decode variants (`IE_DS41_FP8_PACKED` 0-3, all bit-identical per docs/47).
2. **`ds41_hc_mixes` at `T > 1` falls to the ~700 µs/call general kernel** ⇒ ~56-72 ms/step (C.1
   item 16). Must be generalised; doing so is also what buys P2 its bit-identity.

**Link time at `T` rows.** Three calibrations, because this is where the answer lives:
*pessimistic* — the 78/22 imbalance persists at 26 GB/s on the slow link ⇒ 33.3 GB/s effective
aggregate; *central* — 45 GB/s (86% of the 52.4 GB/s pair ceiling of docs/39:95; the split evens out
by the law of large numbers once a layer has 17 misses instead of 1.8, but host-memory contention
and per-expert DMA setup remain); *optimistic* — 52.4 GB/s **and** the 40% correlation discount.
For reference, the *measured* `T = 1` point is 1.285 GB / 56.5 ms = **22.7 GB/s effective** — the
E[max] imbalance plus the EP helper's serialisation (see F.2).

**Drafter cost (Derived): ~8 ms**, flat in `T` (it always drafts five positions; the scheduler only
changes how many are verified). Its dense set is 0.71 GB → 1.7 ms at ~400 GB/s; its 9 routed
experts (3 layers × 3) are 169 MB → **0.5 ms if the 7.22 GB expert pool is VRAM-resident, 6.5 ms if
it is fetched over the link**; the tied head is one pass over the 1.32 GB BF16 head = 2.2 ms; the
five sequential Markov steps read 5 × 66.19 MB = 331 MB ≈ 0.6 ms plus five launch/sync round trips
≈ 1 ms; six `hc_mixes` calls at the 5-row shape. **8 ms assumes a resident drafter pool.**

**Central case (45 GB/s, independence), `pass(T) = dense(T) + link(T) + 8`:**

| T | link bytes | link ms | dense ms | pass ms | max tokens | **ms/token at full acceptance** | **tok/s** |
|---|---|---|---|---|---|---|---|
| 1 (today) | 1.285 GB | 56.5 (measured) | 31.6 | 88.1 | 1 | 88.1 | **11.3** |
| 2 | 3.652 GB | 81 | 33.9 | 123 | 2 | 61.5 | 16.3 |
| 3 | 5.934 GB | 132 | 36.2 | 176 | 3 | 58.7 | 17.0 |
| 4 | 8.195 GB | 182 | 38.5 | 229 | 4 | 57.3 | 17.5 |
| 5 | 10.435 GB | 232 | 40.8 | 281 | 5 | 56.2 | 17.8 |
| 6 | 12.658 GB | 281 | 43.1 | 332 | 6 | **55.3** | **18.1** |

**Per accepted token at `k = 5` (six rows verified), by acceptance length `L`:**

| L (tokens committed per pass) | pessimistic 33 GB/s | **central 45 GB/s** | optimistic 52.4 GB/s + 40% correlation |
|---|---|---|---|
| 2 | 218 ms → 4.6 tok/s | 166 ms → 6.0 | 104 ms → 9.7 |
| 3 | 145 → 6.9 | 111 → 9.0 | 69 → 14.5 |
| 4 | 109 → 9.2 | 83 → 12.0 | **52 → 19.3** |
| 5 | 87 → 11.5 | 66 → 15.1 | 41 → 24.2 |
| 6 (every draft accepted) | 73 → 13.8 | **55 → 18.1** | 35 → 29.0 |

**Where 20 tok/s lands (Derived):**

* **Pessimistic: never.** Full acceptance gives 13.8 tok/s.
* **Central: never.** Full acceptance gives 18.1 tok/s — DSpark's ceiling on this hardware, at
  `T = 6`, is **1.6×**, and it needs perfect acceptance to get there.
* **Optimistic: `L ≈ 4.1` of 6**, i.e. a per-position acceptance of about **0.85** (with a uniform
  per-position rate `a`, `E[L] = 1 + Σ_{j=1..5} a^j`: `a = 0.7 → 2.94`, `a = 0.8 → 3.69`, `a = 0.85
  → 4.16`, `a = 0.9 → 4.69`).

**The per-token curve is nearly flat** (61.5 → 55.3 ms across `T = 2…6` at full acceptance), which
is itself a finding: the report's confidence scheduler has almost nothing to optimise on this
hardware, because the fetch scales ~linearly with the verification length while only the dense body
amortises. The optimum is weakly determined and a fixed `k = 5` is a reasonable first choice.

### D.4 The levers that change this arithmetic, in order of size

1. **Even the link split and de-serialise the EP helper (a prerequisite, not an option).** At
   `T = 1` the effective aggregate is 22.7 GB/s against a 52.4 GB/s pair ceiling. Two causes, both
   **Derived from docs/47's columns, neither Verified by a run**: (a) the E[max] imbalance — the
   slower card carries 78% of the misses, so an even split would cut its 39.7 ms to 25.5, worth
   **~14 ms/token**; (b) the remote tier's 36.1-36.8 ms wall against a 19.4 ms tail means the
   helper's effective start slips ~0.5 ms per layer (thread spawn + `ep_hx` staging + queue
   submission) — docs/47 term 1 confirms the tail *is* the wait for the remote card ("the tail
   shrank 20.6 → 17.3 as the owner finished later and waited less"), so **~19 ms/token** is
   recoverable serialisation. Together **~20-31 ms/token at `T = 1` alone ⇒ 14-17 tok/s with no
   speculation whatsoever**, and the same fix is what makes the `T = 6` link estimate 45 rather than
   33 GB/s. **Cheaper than DSpark, and DSpark needs it anyway.**
2. **The CPU miss split, which `T > 1` transforms.** At `T = 1` each expert serves one row, so the
   CPU reads 18.8 MB from the pinned arena for 3 GEMVs — terrible intensity, and
   `IE_DS41_QSTAR = 0.30` reflects that. **At `T = 6` each fetched expert serves ~1.1 rows on
   average but the *union* is read once for all six rows, and the CPU reads it from host memory at
   ~45-66 GB/s instead of 26 GB/s over a link.** Every expert computed on the CPU removes 17.93 MiB
   from the saturated link. With 12 E-cores and the measured 45-66 GB/s of host bandwidth, the CPU
   could plausibly absorb 20-35% of the misses — **directly off the critical path**. The machinery
   exists (`CpuMiss`, deepseek41_experts.hpp:203-221) but is `T = 1` only and its EP-export path is
   broken (docs/47: "in export mode the remote tier's `y` is null … the path had never been
   runnable"). **This may be the strongest lever DSpark unlocks.**
3. **Grow the static tier.** Every static slot per layer per card removes ≈ `p_e · T` misses.
   Phase 19's `o_a` in FP8 frees 1.3 GB (≈ 1.7 slots/layer/card) and the head in Q8 frees 0.66 GB
   (≈ 0.9 slots) → static mass 2.908 → ~3.25 per token, ~8% fewer misses at `T = 6`. Real but
   second-order.
4. **The drafter's 7.22 GB expert pool: make it fully static.** 128 × 3 slots at 18.80 MB. Cost:
   7.22 GiB of VRAM ≈ 9.6 backbone static slots per layer summed across both cards (≈ 4.8 per card
   per layer, against today's 22/21) → the backbone's static hit rate falls and its miss bytes rise
   ~8%. Benefit: the drafter costs 8 ms instead of ~14 ms per pass, **and** the drafter's fetch
   would otherwise contend with the verify's fetch on the same saturated link. **Recommendation:
   static.** (The alternative — the drafter's 9 experts per pass over the link — adds 169 MB to a
   link that is the binding constraint.)
5. **Bytes per expert are irreducible.** Of 17.93 MiB, 1.05 MiB (5.9%) is the E8M0 scale plane and
   the rest is already FP4. No lever here short of another quantisation step.
6. **A shorter verification length.** Given the flat curve, worth little on this hardware. Keep
   `k = 5`.

---

## E. The phase plan

Campaign style: **criteria written before the build, one bounded change per phase, a gate per
phase, opt-in switches until the gate passes.** Every phase below names its pass criteria.

### P0 — Measure the union, before anything is built (read-only, no new kernel)

**This phase decides whether the campaign runs at all.** Section D's answer swings from 13.8 to
29.0 tok/s on two unmeasured quantities; both can be read off the committed binary.

*Build:* nothing. A small tool (or an `IE_DS41_DUMP_ROUTING` env on the existing decode test) that
records, per decode step and per layer, the `h_idx` array the forward already copies to the host at
forward.cpp:816, plus `Ds41ExpertTier::tier_of(L, e)` for each selected expert.

*Pass criteria:*
1. Over ≥ 200 consecutive decode steps on real text at a 2,048-token context, report for each
   window size `T ∈ {1..6}` of **consecutive** steps: the mean union size per layer, and its split
   into static / pinned / mmap by `tier_of`. **Compare against D.2's table.** Report the implied
   link bytes per pass and per token.
2. Report the same for the *same number* of independently drawn steps (shuffle the step order) —
   the difference between consecutive and shuffled windows **is** the routing correlation, measured
   rather than assumed.
3. From the per-step `Ds41LayerStats`, report the per-layer owner/remote miss split and the
   `moe_call_ms` / `moe_remote_ms` / `ms_tail` triple, and state whether the remote tier overlaps
   the owner's groups (lever D.4.1(b)).

*Gate decision:* if measured `T = 6` misses/layer ≥ 15 (i.e. the independence model holds), the
central column applies and **DSpark's ceiling is ~18 tok/s at perfect acceptance** — the campaign
should then be re-scoped around levers D.4.1 and D.4.2 first, with DSpark as a multiplier on top.
If measured misses/layer ≤ 11, the optimistic column applies and DSpark can reach 20 at a per-
position acceptance of ~0.85. **Report the number; let the founder choose.**

### P1 — Bind, upload and run the drafter forward against a Python golden

*Build (bounded):* the six `Ds41Dspark` tensor binds + removing the matching `mtp_extra` clauses;
upload for those six + `upload_layer(40..42)`; a second `Ds41ExpertTier` over layers 40-42 with
`E = 128, TK = 3`; three MTP `LayerState` rings; the `main_hidden` capture kernel at layers 37-39;
`ds41_dspark_forward(main_hidden, input_ids, pos0)` → `(output_ids[6], logits[5, V],
confidence[5])`, greedy (`temperature = 0`) only.

*Golden:* extend `tools/ds41_reference/` to dump, for **forced** inputs (a fixed `main_hidden` and
a fixed `input_ids`), per MTP stage: `main_x`, the post-`hc_pre` attention input, the attention
output, the MoE input and output, the block output; then `logits[5, V]`, `output_ids[6]`,
`confidence[5]`. Forced routing indices and weights per stage, as Phases 5-7 did.

*Pass criteria:*
1. Every stage boundary within Phase 7's bars (the same digits the backbone's goldens use), with
   forced routing, at three `pos0` values: `pos0 < 128` (a partly filled ring), `pos0 = 128`
   (exactly filled) and `pos0 = 1000` (wrapped).
2. `output_ids[1..5]` **identical** to the golden's (a discrete outcome — an index mismatch is a
   hard failure), and `confidence[5]` within 1e-3 relative.
3. The drafter's own router flips counted and reported per stage, as Phase 7 does; a flip at a
   large score margin is a defect.
4. The 5-row block is **not** reproducible by five 1-row calls — a negative control that asserts
   the non-causal block mask is actually non-causal (A.3 fact 2).
5. Prefill seeding: after a ≥ 256-token prefill with `bounded_replay` on, the three rings read back
   equal the golden's `window_kv_cache` slot by slot at 3e-3, **and** the segment-to-slot alignment
   is asserted explicitly (C.3).
6. Device bytes: the drafter's resident set reported and compared with 7.933 GB; the static-slot
   loss per backbone layer per card reported.

### P2 — The backbone's decode forward at T > 1, bit-identical to T one-row steps

*Build (bounded):* C.1 items 1, 2, 3, 4, 6, 7, 8, 10, 12, 13, 16, 18. **Opt-in
`IE_DS41_DECODE_MULTI=1` until the gate.** No speculative loop yet.

*Pass criteria:*
1. **The exact-path golden: a `T`-row step at `pos0` must equal `T` consecutive 1-row steps from
   the same state, BIT-FOR-BIT, for `T ∈ {2, 3, 6}` at `pos0 ∈ {even, odd}` (the ratio-2 parity)
   and at `pos0` where the ring wraps.** Compared: all `T` rows of logits; `read_state(L, ...)` for
   every layer (ring slot by slot, latents, index keys, `nc`, `part_valid`); `last_hashes()` as
   integers. Bit-identity — not a bar — is reachable because (a) the ring write, the compressor and
   the index keys are per-row functions of per-row inputs, (b) the T-row chunked `hc_mixes` chunks
   each row exactly as the 1-row shape does, and (c) the attention mask is causal. **If any term
   cannot be made bit-identical, the criterion falls back to a STATED bar with the reason and the
   measured residual — never to silence.**
2. The negative control: a `T = 6` step with item 12's per-row ring causality **disabled** must
   FAIL criterion 1 (proving the criterion has teeth).
3. The item-8 latent RoPE bug is demonstrated on the old code path (a `T = 2` ratio-1 step with the
   single-row table) and fixed.
4. Prefill unchanged: `ds41_forward_test`, `ds41_resident_test`, `ds41_replay_test` digits identical
   to the pre-phase build.
5. Measured: the `T = 6` step's ms, its column split, and its link bytes — **compared against
   D.3's central row (332 ms, 12.658 GB)**. Report which of the two traps (FP8 dequant, `hc_mixes`)
   was the larger term before its fix.

### P3 — Rollback

*Build (bounded):* the snapshots of C.2 (ring slots, `part_*`, the ratio-2 rows' `ckv/cg`) and
`Ds41Forward::rollback_to(uint32_t n_pos)`.

*Pass criteria:*
1. For every `L ∈ {1 … 6}`: run a `T = 6` step from state `S`, `rollback_to(pos0 + L)`, then
   compare the whole state (`read_state` for every layer + `n_pos()` + `all_ids_`) against the
   state reached by running `L` consecutive 1-row steps from `S`. **Bit-identical.**
2. Then continue with a further 1-row step from each rolled-back state and compare its logits
   bit-for-bit against the same step taken on the never-speculated path.
3. The negative control: skipping the ring restore must FAIL criterion 1 for some `L < 6` at
   `pos0 ≥ 128` (it must, by C.2's aliasing argument — if it does not, the argument is wrong and
   that is a finding).
4. Snapshot bytes and per-step snapshot/restore time reported; the criterion is < 1 ms/step.

### P4 — The loop, greedy, lossless

*Build (bounded):* `Ds41SpecGenerator` — draft, verify at `T = 1+k`, greedy acceptance, the bonus
token, rollback, detokenise `L` tokens. Fixed `k = 5`. **Temperature 0 only.** Opt-in
`IE_DS41_SPEC=1`.

*Pass criteria:*
1. **Losslessness, the phase's whole point: for ≥ 5 prompts × ≥ 256 generated tokens at
   temperature 0, the generated sequence must equal plain greedy decode TOKEN FOR TOKEN.** Any
   divergence is a hard failure, not a bar. Run with `IE_DS41_SPEC=0` and `=1` on the same binary
   and `cmp` the id streams.
2. The acceptance histogram reported: the distribution of `L` over ≥ 1,000 passes, the mean, and
   the **per-position acceptance rate** for positions 1-5 separately. **This is the number the
   whole campaign has been waiting for** — compare it against D.3's requirement.
3. The confidence head's calibration measured, not used: for each position, the empirical
   acceptance rate bucketed by the raw confidence scalar, so B.3 step 2's scheduler can be fitted
   later. **Report; do not yet act on it.**
4. Stop conditions, EOS and stop strings behave identically with speculation on and off (the
   `L`-tokens-at-once path must not emit past an EOS that lands mid-block).
5. The existing test set (`ds41_generate_test`, `ds41_decode_test`, `ds41_replay_test`,
   `ds41_resident_test`, `ds41_forward_test`) passes with the switch on **and** off.

### P5 — Measured speed

*Build:* nothing new. Measurement only.

*Pass criteria:*
1. ms/token and tok/s at a 2,048-token context, ≥ 3 samples, against the 88.1-88.3 baseline on the
   same binary with the switch off. Report the step columns, the link bytes per pass and per
   accepted token, the union size per layer, the acceptance histogram, and the drafter's share.
2. The per-`k` curve measured for `k ∈ {1, 2, 3, 5}` — the engine's own version of D.3's table, so
   the report's "profiled engine throughput curves" exist for real.
3. **The verdict stated as a number against 20 tok/s, with the gap attributed term by term.** If
   the number is below 20, the phase does not "fail" — it reports which of D.4's levers the
   measurement says is now the largest.

### Then, in the order D.4 ranks them (each its own phase, criteria first)

P6 the EP split and the helper's serialisation (D.4.1) — the largest lever, and it applies at
`T = 1` too. P7 the CPU miss split at `T > 1` with the export path redesigned (D.4.2; docs/47
already says "Phase 20 needs the CPU results exported as rows, not added into `y` — a design
change"). P8 the small-`M` FP8 GEMV (`M ≤ 8`), so the dense body stays at 1 byte/weight. P9 the
confidence scheduler, fitted on P4 criterion 3's calibration. P10 speculative *sampling* at
temperature > 0, with its own losslessness criterion.

---

## F. Open questions and risks

**F.1 — The union's miss growth. THE risk.** D.2's independence model says misses/layer/token rise
57% from `T = 1` to `T = 6`, which caps DSpark at ~1.6× and ~18 tok/s even at perfect acceptance.
The 40% correlation discount that would make 20 tok/s reachable at `L ≈ 4.1` is a **guess calibrated
by nothing.** *The one measurement that settles it:* P0 — dump `h_idx` per layer over consecutive
decode steps (already copied to the host at forward.cpp:816), compute the union and its `tier_of`
split for consecutive vs shuffled windows. Read-only, no new kernel, no GPU program.

**F.2 — Is the EP helper actually overlapping?** docs/47's columns (groups 38.7 + tail 19.4 = 58.1
against a remote wall of 36.1-36.8) are consistent with the remote card starting ~0.5 ms late per
layer, i.e. **~19 ms/token of recoverable serialisation** — more than the entire dense body. docs/47
term 1's observation that "the tail shrank 20.6 → 17.3 as the owner finished later" says the tail
*is* the wait, so the term is real; its *size* is **Derived from columns, not Verified by a run.**
*The measurement:* per-layer timestamps of the helper's spawn, its queue submission and its
completion against the owner's groups, on one decode bench. **This is cheap and it may be worth more
than the whole speculative build.**

**F.3 — The `T = 6` attention kernel's cost.** The +1.6 ms/row estimate is the weakest term in
D.3's dense model. At `T = 1` the kernel runs 175 µs/call against its own 80-90 µs bench
(docs/47 leaves this "unexplained"), so the marginal row *should* be nearly free — but if the
kernel instead scales linearly in `T`, the dense body goes 31.6 → 66 ms and the `T = 6` pass to
~355 ms (17.0 tok/s at full acceptance). *The measurement:* run `ds4_attention_segs` at `T ∈ {1, 6}`
with the decode shapes (`n_a = 128`, `n_b = 2048`, 64 heads, head_dim 512) in the existing
`deepseek4_attn` bench. No model load needed.

**F.4 — The acceptance length is entirely unknown.** The tech report gives no figure, the reference
implements no loop, and the DSpark paper is not on disk. Everything from "DSpark is worth 1.5×" to
"DSpark is worth 3×" is consistent with what is in the tree. *The measurement:* P4 criterion 2.
There is **no way to shortcut it** — it needs the drafter running against the real backbone. A
cheaper partial proxy: run the *reference* `forward_spec` in PyTorch against the reference backbone
on a few hundred tokens and count greedy matches. That needs the 355 GB checkpoint in a torch
process and a GPU, so it is not free either, but it would answer F.4 before P2-P4 are built.

**F.5 — The confidence head's scale.** `confidence` is a raw fp32 linear output with no activation
(1097). Whether it is a logit (⇒ `σ`), a log-probability, or something else is **Unknown** — the
report says "predicts per-position conditional acceptance probabilities" but the code applies no
squashing. *The measurement:* P4 criterion 3's empirical calibration.

**F.6 — Does the drafter's static pool starve the backbone?** Making the drafter's 7.22 GB pool
static costs ≈ 9.6 backbone static slots per layer across the cards, raising the backbone's miss
bytes ~8% — which at `T = 6` is ~1 GB more over the already-saturated link per pass. *The
measurement:* P1 criterion 6 reports the slot loss; P5 measures whether it nets out.

**F.7 — The ratio-2 compressor at `T > 1` (C.1 item 7) is the highest-risk single change.** It
spans state (`part_*`) across a step, has a parity dependence on `pos0`, and P3's rollback must undo
it exactly. *Mitigation:* P2 criterion 1 tests both parities and P3 criterion 1 tests every `L`;
the item-8 latent RoPE bug (a single-row table used for `n_new > 1` rows) shows this area is
already fragile and unexercised.

**F.8 — Losslessness at temperature > 0 is out of scope and must be said so.** Greedy comparison is
exact only at temperature 0. A user running the server at temperature 0.6 with `IE_DS41_SPEC=1`
would get a *different distribution*, not a different sample — a silent correctness change. **The
switch must refuse `temperature > 0` until P10 lands.** This is a hard rail, not a preference.

**F.9 — Two cards, one of them down.** docs/47 records card 1 (PCI 0000:09:00.0) in a reset loop
with `drm_sched_job_timedout` and every later load returning `OUT_OF_DEVICE_MEMORY`; **a reboot is
needed before any two-card run.** Nothing in this brief was measured, so nothing here is affected,
but P0's dump and every phase after it need both cards.

**F.10 — What this document did not establish.** No build, no run, no GPU. Every ms in section D is
arithmetic over docs/46 and docs/47's published columns plus the checkpoint's byte counts. The three
numbers that would make it a decision rather than an estimate are F.1, F.2 and F.4 — and two of the
three are read-only.
