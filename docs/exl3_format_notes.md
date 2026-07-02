# EXL3 format notes (decoded from exllamav3 source, 2026-06-19)

Captured by reading `turboderp-org/exllamav3` (cloned at `~/exllamav3-ref`, no GPU used).
This is the executable spec for the EXL3 engine-support plan
(`docs/superpowers/plans/2026-06-19-qtip-quant-engine.md`), Tasks 1 (host decode),
2 (SYCL kernel + Hadamard), 4 (importer).

**EXL3 = a QTIP variant** (procedural codebook + tail-biting trellis; differs from QTIP in
regularization + packing). Same SOTA quality class. Source of truth:
`exllamav3/exllamav3_ext/quant/{exl3_dq.cuh,codebook.cuh}` (decode) and
`exllamav3/modules/quant/exl3.py` + `exl3_lib/quantize.py` (packing/storage).

## 1. What's stored per Linear (safetensors → the importer, Task 4)
From `exl3.py` `LinearEXL3`, the per-layer keys are `<key>.{trellis,suh|su,svh|sv}` (+ optional `mcg`, `mul1`):
- **`trellis`** — `int16`, the packed trellis codes (the quantized weight bitstream). `K = trellis.shape[-1] // 16`. Packed in **256-weight blocks**, `bits` bits/weight (1–8 bpw), tail-biting trellis. Each 256-block occupies `bits * 256` bits = `bits * 8` `uint32` words.
- **`suh`** — `half`, input-side (per-`in_feature`) Hadamard sign/scale vector. (Or `su` = `int16` packed → unpack to `suh`.)
- **`svh`** — `half`, output-side (per-`out_feature`) sign/scale vector. (Or `sv` packed.)
- **`mcg`**, **`mul1`** — optional alternative codebook multiplier constants (default codebook `cb=0` uses the hardcoded MCG below; if `mcg`/`mul1` present, use those).
- `in_features` (K), `out_features` (N). **Note:** "scale is no longer used" — suh/svh carry all scaling.

## 2. Code extraction (bitstream → 16-bit word), from `exl3_dq.cuh`
For weight index `t_offset` in a 256-block, the 16-bit code is the `bits`-wide field at bit
offset `t_offset * bits` (tail-biting: indices wrap mod `bits*256` within the block; the kernel
loads two `uint32`s and funnel-shifts to extract the 16-bit `w`). Per-`bits` aligned fast paths
exist (`dq8_aligned_{1,2,4}bits`, `dq{2,4,8}`), but the canonical extraction is:
`w = (funnelshift(b, a, s)) & 0xFFFF` where `a,b` are the two `uint32` words straddling the field.

## 3. The decode — `decode_3inst<cb=0>` (the computed code), from `codebook.cuh`
**Pure ALU, no lookup table** (the Arc-friendly property). For the default codebook `cb=0`:
```
x  = w * 89226354u;                     // MCG hash step
x += 64248484u;
x  = (x & 0x8fff8fff) ^ 0x3b603b60;     // CUDA lop3 lut 0x6a == (a & b) ^ c
// reinterpret x as two fp16 (low half, high half):
val = f16(x & 0xffff) + f16(x >> 16);   // sum of two halves → pseudo-Gaussian sample
```
That's the entire procedural codebook for `cb=0`. Three more codebooks exist (`cb=1` uses
multiplier `0xCBAC1FED`; `cb=2` uses `0x83DCD12D` + a `vabsdiff4` popcount-ish step) — start
with `cb=0` (the common case). **SYCL port is direct:** `lop3 0x6a` → `(x & 0x8fff8fff) ^ 0x3b603b60`;
`half2_uint32` → `sycl::bit_cast`/`reinterpret` of `uint32` to `sycl::half[2]`; `__hadd` → `+`.

## 4. The matmul-level reconstruction (forward — Task 2 kernel + Hadamard)
EXL3 stores **Hadamard-incoherence-rotated** weights, so the runtime rotates the *activation*
instead (QuaRot-style). Per linear `y = x · W`:
1. `xh = hadamard(x) ⊙ suh`  — Walsh–Hadamard transform on the input activation (block size = the EXL3 Hadamard size), then multiply by the input sign/scale vector `suh`. (`preapply_had_l/_r`, `had_k`, `had_n` in `exl3_lib/quantize.py`.)
2. `acc = xh @ decode_3inst(trellis)` — GEMV/GEMM against the trellis-decoded weights `[K,N]`.
3. `y = acc ⊙ svh` — scale output by the output sign/scale vector.

So the engine needs **(a)** a Walsh–Hadamard transform kernel on activations, **(b)** the trellis
decode + MAC (`gemv_exl3`), **(c)** the suh/svh elementwise scales. (a) is new; (b) mirrors
`gemv_q4_K`'s decode-then-MAC shape; (c) is trivial.

## 5. Validation / oracle — RESOLVED 2026-06-19 (CUDA-free)
exllamav3's decode is CUDA-ext-only (no pure-torch/numpy path exists anywhere in the repo —
confirmed across `tests/`, `eval/`, `convert.py`, `util/`). **But the decode is fully reproducible
on CPU in ~30 lines of numpy** from §2–§3 + §7 below — no GPU, no CUDA. **Oracle strategy:**
1. **numpy reference decoder** (CPU): load a downloaded `trellis`/`suh`/`svh` via the `safetensors`
   python lib (numpy backend, no torch), decode per §2–§3 + the tensor-core un-permute (§7), apply
   the 128-Hadamard + suh/svh (§4/§7) → golden `[K,N]` fp16. This is the ground truth the C++ host
   decode (`exl3_decode_host`, Task 1) is tested against.
2. **The one empirically-risky bit is hand-verified:** `lop3.b32 lut 0x6a` = `(a & b) ^ c` for all
   8 (a,b,c) combos (truth table 0b01101010) → `x = (x & 0x8fff8fff) ^ 0x3b603b60`. No CUDA needed.
3. **Ultimate semantic check:** end-to-end PPL on the small model (`ie-perplexity` EXL3 vs Q4_K) — a
   wrong decode/permutation/Hadamard yields garbage weights → gibberish, so coherent text + PPL ≤
   Q4_K confirms the whole pipeline. (Optional extra-confidence cross-check: reconstruct(EXL3) vs the
   ORIGINAL fp16 model weights within quant tolerance — real independent ground truth, but needs the
   ~2.5 GB fp16 model; skip unless the numpy↔C++↔PPL chain disagrees.)

**Smallest artifact:** `turboderp/Llama-3.2-1B-Instruct-exl3` @ **2.0bpw ≈ 0.8 GB** (per-bpw HF
branches). Spec `eval/spec/*.json` point at the author's local paths — real files are under the
`turboderp` HF org.

## 7. VERIFIED corrections + additions (source-read pass 2, 2026-06-19)
Authoritative re-read of exllamav3 (file:line). Supersedes any conflict above.
- **Storage shapes:** `trellis` int16 `(K/16, N/16, 16*bits)` (3-D, asserted `exl3.py:44`);
  `suh` fp16 `(K,)` = per-INPUT sign+scale; `svh` fp16 `(N,)` = per-OUTPUT sign+scale. `bits` is
  NOT stored — derived `bits = trellis.shape[-1] // 16` (`exl3.py:57`). Optional `bias` fp16 `(N,)`.
  Legacy `su`/`sv` (int16 sign bitfields) only in old files → unpack to suh/svh.
- **Codebook selection (`reconstruct.cu:127-129`):** `cbi = bits-1`; `+8` if a `mcg` tensor present
  (cb=1, mult `0xCBAC1FED`, no add), `+16` if `mul1` present (cb=2, different recon: vabsdiff4 +
  FMA). **Default = cb=0** (neither tensor) — what downloaded turboderp models use. Implement cb=0
  first; treat presence of `mcg`/`mul1` keys as the switch.
- **cb=0 decode (`codebook.cuh:25-35`):** `x = (x*89226354u + 64248484u); x = (x & 0x8fff8fff) ^
  0x3b603b60; w = f16(x & 0xffff) + f16(x >> 16)`. The `codebook_scale = 1.24371088`
  (`quantize.py:15`) is FOLDED into suh at quant time — do NOT re-apply it.
- **Bit extraction (`exl3_dq.cuh:15-31`) — TAIL-BITING per 16×16 tile:** for weight `t` in a tile,
  `b0 = t*bits + bits - 16 + 256*bits`, `i0=b0/32`, `i1=(b0+15)/32`, `s0=(i1+1)*32-(b0+16)`, read
  `a=u32[i0 % (bits*256/32)]`, `b=u32[i1 % (bits*256/32)]`, `code = funnelshift_r(b,a,s0) & 0xffff`
  (low word = b = u32[i1], high = a = u32[i0]). The `%` is the cyclic wrap within the tile.
- **⚠ NEW — TENSOR-CORE TILE PERMUTATION (`quantize.py:21-48`):** the 256 codes decoded from one
  tile are in the kernel's lane/interleave order (warp lane `l*8 + (0..7)`), NOT row-major. The host
  oracle MUST apply the inverse of `tensor_core_perm` (pure Python, `perm_i = argsort(perm)`) to lay
  the 16×16 tile out row-major before assembling `W[K,N]`. Missing this = silently-shuffled weights.
- **Hadamard (`hadamard.cu:107`, `hadamard_inner.cuh`):** fixed **128-point Sylvester** (power-of-two
  butterfly, hard-wired — NOT the `hadamard_data/*.txt` Paley matrices, which are for odd dims used
  elsewhere). Norm **`1/sqrt(128) = 0.088388347648`**. Applied BOTH sides at quant (left over K, right
  over N); at inference applied to the activation in / output out (orthogonal ⇒ algebraically same).
  Requires K,N multiples of 128. Forward order (`exl3.py:150-167,184-190`):
  `xh = had128(x) ⊙ suh ; acc = xh @ decode(trellis) ; y = had128(acc) ⊙ svh`.

## 6. Direct portability verdict
The decode is `bits`-field extraction + a 3-instruction integer hash + a 2×fp16 reinterpret-and-sum.
None of it needs CUDA intrinsics that lack SYCL equivalents. The only genuinely new kernel is the
Hadamard transform on activations. The risk remains where the plan says: mapping the parallel
trellis walk onto Arc subgroup width (Task 2) — but the per-weight decode itself is independent
(each `t_offset` decodes from its own bit field), so it parallelizes cleanly.
