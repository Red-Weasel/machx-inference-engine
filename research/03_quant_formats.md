# Quantization Formats — Byte-Level Reference for the Inference Engine

> Target audience: someone writing GPU dequant kernels by hand against this document.
> Date: 2026-04-25. All struct definitions verified against `ggml-common.h` HEAD,
> `ggml-quants.c` HEAD, AutoAWQ `awq/utils/packing_utils.py` and `awq/modules/linear/gemm.py`,
> AutoGPTQ `qlinear_cuda.py`, IST-DASLab `marlin/__init__.py`, intel/auto-round
> `auto_round/export/export_to_autoround/export.py`. URLs cited inline.

---

## 1. GGUF Container

GGUF is a single-file, mmap-friendly container produced by `llama.cpp`.
It stores a key/value metadata header followed by a tensor-info table and a tensor-data blob.

### 1.1 Magic, version, alignment

| Field | Bytes | Value |
|---|---|---|
| Magic | 4 | `0x47 0x47 0x55 0x46` (ASCII `"GGUF"`, little-endian word `0x46554747`) |
| Version | 4 | `uint32_t`, currently `3` (v3 added BE support) |
| Tensor count | 8 | `uint64_t` |
| Metadata KV count | 8 | `uint64_t` |

After the fixed 24-byte header come the KV pairs, then the tensor-info table, then
zero padding up to `general.alignment` (default **32**, must be multiple of 8),
then the tensor data blob.

The padding rule used everywhere in GGUF:

```
pad(off) = off + ((ALIGN - (off % ALIGN)) % ALIGN)
```

Tensor offsets in the info table are **relative to the start of the tensor-data section**,
not the file. Each tensor offset is a multiple of `general.alignment`.

Source: `ggml/docs/gguf.md` (HEAD).

### 1.2 KV-pair encoding

```
struct gguf_string_t { uint64_t len; uint8_t data[len]; }       // UTF-8, no NUL
struct gguf_kv_t {
    gguf_string_t key;          // lower_snake_case dotted, ≤ 65535 bytes
    uint32_t      value_type;   // gguf_metadata_value_type
    /* value follows, encoding depends on value_type */
};
```

`value_type` enum (subset I actually use):

| name | id |
|---|---|
| UINT8 | 0 |
| INT8 | 1 |
| UINT16 | 2 |
| INT16 | 3 |
| UINT32 | 4 |
| INT32 | 5 |
| FLOAT32 | 6 |
| BOOL | 7 |
| STRING | 8 |
| ARRAY | 9 |
| UINT64 | 10 |
| INT64 | 11 |
| FLOAT64 | 12 |

`ARRAY` payload: `uint32_t inner_type, uint64_t n, inner_type values[n]`.

### 1.3 Tensor info entry

```c
struct gguf_tensor_info_t {
    gguf_string_t name;             // ≤ 64 chars
    uint32_t      n_dimensions;     // ≤ 4 (current GGML_MAX_DIMS)
    uint64_t      shape[n_dimensions];
    uint32_t      type;             // ggml_type enum (next section)
    uint64_t      offset;           // bytes from start of tensor_data section
};
```

### 1.4 `ggml_type` enum (verified vs `ggml.h` HEAD)

```
F32        = 0       Q5_K       = 13      I64        = 27
F16        = 1       Q6_K       = 14      F64        = 28
Q4_0       = 2       Q8_K       = 15      IQ1_M      = 29
Q4_1       = 3       IQ2_XXS    = 16      BF16       = 30
/* 4,5 removed */    IQ2_XS     = 17      /* 31..33 reserved */
Q5_0       = 6       IQ3_XXS    = 18      TQ1_0      = 34
Q5_1       = 7       IQ1_S      = 19      TQ2_0      = 35
Q8_0       = 8       IQ4_NL     = 20      /* 36..38 reserved */
Q8_1       = 9       IQ3_S      = 21      MXFP4      = 39
Q2_K       = 10      IQ2_S      = 22      COUNT      = 40
Q3_K       = 11      IQ4_XS     = 23
Q4_K       = 12      I8/I16/I32 = 24/25/26
```

Note: ids 4 and 5 are retired (Q4_2 / Q4_3). My loader rejects them.

### 1.5 ~40-line minimal C parser (the part you actually need)

```c
/* GGUF parser sketch — assumes little-endian host, file is mmap'd at base. */
typedef struct { uint64_t len; const char *data; } gstr_t;
static const uint8_t *rd_str(const uint8_t *p, gstr_t *s) {
    memcpy(&s->len, p, 8); s->data = (const char*)(p + 8); return p + 8 + s->len;
}
static const uint8_t *skip_val(const uint8_t *p, uint32_t t);  /* fwd */
static const uint8_t *skip_val(const uint8_t *p, uint32_t t) {
    static const uint8_t sz[13] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
    if (t < 8 || t == 10 || t == 11 || t == 12) return p + sz[t];
    if (t == 8) { gstr_t s; return rd_str(p, &s) + s.len - 8 + 8; } /* see rd_str */
    if (t == 9) {
        uint32_t it; uint64_t n; memcpy(&it,p,4); memcpy(&n,p+4,8); p+=12;
        if (it == 8) { gstr_t s; for (uint64_t i=0;i<n;i++) p = rd_str(p,&s); return p; }
        return p + n * sz[it];
    }
    abort();
}
int gguf_parse(const uint8_t *base, size_t len) {
    if (memcmp(base, "GGUF", 4)) return -1;
    uint32_t ver; memcpy(&ver, base+4, 4); if (ver != 3) return -2;
    uint64_t n_tensor, n_kv;
    memcpy(&n_tensor, base+8, 8); memcpy(&n_kv, base+16, 8);
    const uint8_t *p = base + 24;
    uint64_t align = 32;                                /* default */
    for (uint64_t i = 0; i < n_kv; i++) {
        gstr_t k; p = rd_str(p, &k);
        uint32_t t; memcpy(&t, p, 4); p += 4;
        if (k.len == 17 && !memcmp(k.data,"general.alignment",17) && t == 4)
            memcpy(&align, p, 4);
        p = skip_val(p, t);
    }
    /* tensor info table */
    for (uint64_t i = 0; i < n_tensor; i++) {
        gstr_t name; p = rd_str(p, &name);
        uint32_t nd; memcpy(&nd, p, 4); p += 4;
        uint64_t shape[4]; memcpy(shape, p, 8*nd); p += 8*nd;
        uint32_t type; uint64_t off;
        memcpy(&type, p, 4); p += 4;
        memcpy(&off, p, 8); p += 8;
        /* register tensor (name, type, shape, off) */
    }
    /* tensor data starts at next align boundary */
    size_t cursor = (size_t)(p - base);
    cursor += (align - (cursor % align)) % align;
    /* tensors live at base + cursor + tensor.off */
    return 0;
}
```

---

## 2. K-quants — exact struct layouts and dequant formulas

All K-quants use a **super-block of 256 elements** (`#define QK_K 256`) and 8
sub-blocks of 32 elements. Source: `ggml/src/ggml-common.h`.

### 2.1 `block_q8_0` — legacy 32-element 8-bit, **34 bytes**, 8.5 bits/w

```c
#define QK8_0 32
typedef struct {
    ggml_half d;        /* offset 0  : 2 bytes, FP16 scale  */
    int8_t    qs[32];   /* offset 2  : 32 bytes, signed Q   */
} block_q8_0;           /* sizeof = 34, no padding         */
```

Dequant: `y[i] = d * qs[i]`. Trivial. This is the format you use for activations
and as a high-quality fallback.

### 2.2 `block_q4_K` — **144 bytes**, 4.5 bits/w

```c
#define QK_K          256
#define K_SCALE_SIZE   12
typedef struct {
    /* offset 0 : 4 bytes, super-block scales (FP16 pair) */
    union { struct { ggml_half d, dmin; }; ggml_half2 dm; };
    /* offset 4  : 12 bytes, 8 × (6-bit scale, 6-bit min) packed -- see 2.6 */
    uint8_t scales[K_SCALE_SIZE];
    /* offset 16 : 128 bytes, 256 × 4-bit quants (low/high nibble pairs) */
    uint8_t qs[QK_K/2];
} block_q4_K;           /* sizeof = 144 */
```

Dequant (lifted from `dequantize_row_q4_K`, `ggml-quants.c`):

```c
const float d   = fp16_to_fp32(b->d);
const float dm  = fp16_to_fp32(b->dmin);
int is = 0;
const uint8_t *q = b->qs;
for (int j = 0; j < 256; j += 64, q += 32, is += 2) {
    uint8_t sc, m;
    get_scale_min_k4(is + 0, b->scales, &sc, &m);
    float d1 = d * sc, m1 = dm * m;
    for (int l = 0; l < 32; ++l) y[j + l]      = d1 * (q[l] & 0x0F) - m1;

    get_scale_min_k4(is + 1, b->scales, &sc, &m);
    float d2 = d * sc, m2 = dm * m;
    for (int l = 0; l < 32; ++l) y[j + 32 + l] = d2 * (q[l] >>   4) - m2;
}
```

Note that the low and high nibbles of `qs[l]` belong to **different sub-blocks**
(`is+0` and `is+1`). This matters for SLM / register layout; see §7.

### 2.3 `block_q5_K` — **176 bytes**, 5.5 bits/w

```c
typedef struct {
    union { struct { ggml_half d, dmin; }; ggml_half2 dm; };  /* 0  : 4  */
    uint8_t scales[K_SCALE_SIZE];                              /* 4  : 12 */
    uint8_t qh[QK_K/8];                                        /* 16 : 32  -- 5th bits */
    uint8_t qs[QK_K/2];                                        /* 48 : 128 -- low 4    */
} block_q5_K;           /* sizeof = 176 */
```

Dequant:

```c
const float d  = fp16_to_fp32(b->d);
const float dm = fp16_to_fp32(b->dmin);
const uint8_t *ql = b->qs, *qh = b->qh;
uint8_t u1 = 1, u2 = 2;
int is = 0;
for (int j = 0; j < 256; j += 64, ql += 32, is += 2, u1 <<= 2, u2 <<= 2) {
    uint8_t sc, m;
    get_scale_min_k4(is + 0, b->scales, &sc, &m);
    float d1 = d * sc, m1 = dm * m;
    for (int l = 0; l < 32; ++l)
        y[j + l]      = d1 * ((ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - m1;

    get_scale_min_k4(is + 1, b->scales, &sc, &m);
    float d2 = d * sc, m2 = dm * m;
    for (int l = 0; l < 32; ++l)
        y[j + 32 + l] = d2 * ((ql[l] >>   4) + ((qh[l] & u2) ? 16 : 0)) - m2;
}
```

The `qh` byte holds the 5th bit for **eight** values (one per sub-block within
the super-block half). The mask `u1`/`u2` walks across the sub-blocks; this is
the part most people get wrong on first port.

### 2.4 `block_q6_K` — **210 bytes**, 6.5625 bits/w

```c
typedef struct {
    uint8_t  ql[QK_K/2];     /* 0   : 128 -- low 4 bits          */
    uint8_t  qh[QK_K/4];     /* 128 : 64  -- high 2 bits         */
    int8_t   scales[QK_K/16];/* 192 : 16  -- per-16 scale, int8  */
    ggml_half d;             /* 208 : 2   -- super-block scale   */
} block_q6_K;                /* sizeof = 210 */
```

Note Q6_K has **only `d`**, no `dmin` — the 6-bit values are signed (offset by 32).

Dequant:

```c
const float d = fp16_to_fp32(b->d);
const uint8_t *ql = b->ql, *qh = b->qh;
const int8_t  *sc = b->scales;
for (int n = 0; n < 256; n += 128, y += 128, ql += 64, qh += 32, sc += 8) {
    for (int l = 0; l < 32; ++l) {
        int is = l / 16;
        int8_t q1 = (int8_t)((ql[l]      & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
        int8_t q2 = (int8_t)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
        int8_t q3 = (int8_t)((ql[l]      >>   4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        int8_t q4 = (int8_t)((ql[l + 32] >>   4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        y[l +  0] = d * sc[is + 0] * q1;
        y[l + 32] = d * sc[is + 2] * q2;
        y[l + 64] = d * sc[is + 4] * q3;
        y[l + 96] = d * sc[is + 6] * q4;
    }
}
```

A single `qh` byte feeds **four** values (offsets l, l+32, l+64, l+96).
Read pattern: `ql[64]` and `qh[32]` per 128-element half-block.

### 2.5 The 6-bit scale/min interleaving (`get_scale_min_k4`)

Q4_K and Q5_K share a 12-byte `scales[]` field that encodes **eight** pairs of
6-bit values (8 scales + 8 mins = 16 × 6 bits = 96 bits = 12 bytes).
The packing is non-obvious; here is the exact unpack from `ggml-quants.c`:

```c
static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j]     & 0x3F;                                  /* low 6 bits */
        *m = q[j + 4] & 0x3F;
    } else {
        *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>   4) | ((q[j - 0] >> 6) << 4);
    }
}
```

Layout in plain English (12 bytes labelled `q[0]..q[11]`):

| byte | bits 0..5 | bits 6..7 |
|---|---|---|
| q[0] | scale[0] (6 bits) | high 2 bits of scale[4] |
| q[1] | scale[1] | high 2 bits of scale[5] |
| q[2] | scale[2] | high 2 bits of scale[6] |
| q[3] | scale[3] | high 2 bits of scale[7] |
| q[4] | min[0] | high 2 bits of min[4] |
| q[5] | min[1] | high 2 bits of min[5] |
| q[6] | min[2] | high 2 bits of min[6] |
| q[7] | min[3] | high 2 bits of min[7] |
| q[8] | low 4 bits of scale[4] (lo nibble), low 4 bits of min[4] (hi nibble) | |
| q[9] | low 4 bits of scale[5] / min[5] | |
| q[10] | low 4 bits of scale[6] / min[6] | |
| q[11] | low 4 bits of scale[7] / min[7] | |

So scales[0..3] and mins[0..3] are stored straight in the low 6 bits of q[0..7].
Scales[4..7] and mins[4..7] have their **low 4 bits in q[8..11]** (nibble-paired)
and their **high 2 bits stolen from the unused top 2 bits of q[0..7]**.

This is the single trickiest bit of the GGUF k-quant family. On Xe2 I plan to
unpack the 12 bytes into a register file of 8 fp16 scales + 8 fp16 mins once per
super-block and reuse across the 256-element body.

---

## 3. AWQ (Activation-aware Weight Quantization)

AWQ is INT4-only in practice. Two GEMM variants exist (`GEMM`, `GEMV`), with the
same on-disk format; Marlin and BitBLAS variants repack at load time.

Sources: `awq/utils/packing_utils.py`, `awq/modules/linear/gemm.py`.

### 3.1 safetensors keys (per nn.Linear)

| key | shape | dtype |
|---|---|---|
| `*.qweight` | `(in_features, out_features // 8)` | `int32` |
| `*.qzeros`  | `(in_features // group_size, out_features // 8)` | `int32` |
| `*.scales`  | `(in_features // group_size, out_features)` | `float16` |

There is **no `g_idx`** in AWQ — AWQ does not support `act_order`. Group size is
always **128** in practice (`q_group_size: 128, w_bit: 4` in `quant_config.json`).

### 3.2 The AWQ permutation

Both `qweight` and `qzeros` are packed along the **output (n)** dimension: 8
INT4s per INT32. Before packing, the 8-wide chunks are **permuted** by
the constant array

```python
AWQ_ORDER         = [0, 2, 4, 6, 1, 3, 5, 7]
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
```

Source: `awq/utils/packing_utils.py`. The permutation exists so the dequantized
8-wide nibble vector lines up with the way the AWQ GEMM kernel issues
`mma`/`dp4a`-style ops. If you write your own kernel and ignore the permutation,
your weights look like garbage in groups of 8 columns. The permutation is
applied identically to qweight and qzeros — meaning `qzeros[i, k]` decodes to
the zero for output channel `unpack(k)[AWQ_REVERSE_ORDER[lane]]`.

Effective bit-pack of one INT32 cell (low → high bits):

```
[ z_p[0] | z_p[1] | z_p[2] | z_p[3] | z_p[4] | z_p[5] | z_p[6] | z_p[7] ]
   bit 0     4         8       12      16      20      24      28
where p = AWQ_ORDER, i.e. logical output column o ∈ {0..7} lives in nibble
position p^{-1}(o) = AWQ_REVERSE_ORDER[o].
```

### 3.3 Dequant formula

```
group  = in_idx // 128
nibble = AWQ_REVERSE_ORDER[out_idx % 8]
w_q    = (qweight[in_idx, out_idx // 8] >> (4 * nibble)) & 0xF        # 0..15
z_q    = (qzeros [group,  out_idx // 8] >> (4 * nibble)) & 0xF        # 0..15
s      = scales  [group,  out_idx]
w_fp16 = s * (w_q - z_q)
```

There is **no +1 zero offset** in AWQ (that's the GPTQ thing, see below). AWQ
zeros are stored exactly as quantized.

### 3.4 Where to read the kernel

The canonical reference is `awq_ext/quantization/gemm_cuda_gen.cu` in AutoAWQ;
the dequant loop near the top of `gemm_forward_cuda` shows the
shift-by-`4*AWQ_REVERSE_ORDER[i]` and subtract pattern. (Direct GitHub pull of
that file 404'd during research; the file is regenerated by the build, but the
permutation logic mirrors `awq/utils/packing_utils.py`.)

---

## 4. GPTQ

Sources: `auto_gptq/nn_modules/qlinear/qlinear_cuda.py`, `IST-DASLab/marlin/__init__.py`.

### 4.1 safetensors keys (AutoGPTQ vanilla)

| key | shape | dtype |
|---|---|---|
| `*.qweight` | `(in_features // 32 * bits, out_features)` | `int32` |
| `*.qzeros`  | `(ceil(in_features / group_size), out_features // 32 * bits)` | `int32` |
| `*.scales`  | `(ceil(in_features / group_size), out_features)` | `float16` |
| `*.g_idx`   | `(in_features,)` | `int32` |
| `*.bias`    | `(out_features,)` | `float16` (optional) |

For 4-bit: `qweight` is `(in_features/8, out_features) int32` packed along the
**input (k)** dimension — opposite of AWQ which packs along **n**. This is the
single most important thing to remember when porting a GPTQ kernel after writing
an AWQ one.

`g_idx[i] ∈ [0, n_groups)` maps each input row to its group; with `desc_act=True`
this is a non-trivial permutation, otherwise it is just `i // group_size`.

### 4.2 INT4 packing (vanilla GPTQ-for-LLaMA)

```python
# pack: 8 consecutive 4-bit values along k-dim into one int32
for j in range(i, i + 8):
    qweight[row] |= intweight[j] << (4 * (j - i))
```

So nibble layout in one int32, low → high bits, is **k+0, k+1, k+2, … k+7**.
**No permutation.** GPTQ-for-LLaMA stays in natural order.

### 4.3 The famous +1 zero offset

When packing zeros, AutoGPTQ subtracts 1: `zeros = zeros - 1`. The forward path
adds it back: `zeros = zeros + 1`. The reason is that the original GPTQ-for-LLaMA
allowed the encoded zero point to use the full 0..15 range while keeping
`(weight - zero)` representable cleanly; the asymmetric formula reads:

```
w_q   = (qweight[k_packed, n] >> (4 * (k % 8))) & 0xF
z_pkg = (qzeros [g, n_packed] >> (4 * (n % 8))) & 0xF
z_q   = z_pkg + 1                                 # the +1 lives only at decode
s     = scales[g, n]
w_fp16 = s * (w_q - z_q)
```

If you forget the `+1` you get a uniform fp16 bias error of `s` on every weight.

For **symmetric** GPTQ (`sym=True`), zeros are constant 8 (so `z_q = 8` after
the +1 from packed value 7) and qzeros may be omitted; check `quantize_config.json`.

### 4.4 `desc_act` / `act_order`

When `desc_act=True`, GPTQ permutes input columns by descending Hessian
diagonal during quantization. The permutation lives in `g_idx`. Two strategies
on the kernel side:

1. **Apply g_idx on the fly** (vanilla CUDA kernel): scatter-gather scales by
   `g_idx[k]` per inner loop iteration. Slow.
2. **Pre-permute weights at load time** so the kernel runs as if `desc_act=False`,
   then apply the inverse permutation to activations. This is what
   `exllama_v2` and Marlin do.

### 4.5 Marlin layout (`gptq_marlin`)

Marlin reorganises GPTQ weights for Tensor-Core / `mma.m16n8k16`-friendly
access. The packing pipeline (from `IST-DASLab/marlin/__init__.py`):

```python
tile = 16
# Stage 1: tile decompose along k and n
w = w.reshape(k // 16, 16, n // 16, 16).permute(0, 2, 1, 3).reshape(k // 16, n * 16)

# Stage 2: 256-element interleave
#   _perm is built from a base 32-element permutation, reshape (-1, 8),
#   then column-shuffle by [0, 2, 4, 6, 1, 3, 5, 7]
w = w.reshape(-1, _perm.numel())[:, _perm].reshape(orig_shape)

# Stage 3: 8 × 4-bit into uint32 along the n dimension
q = np.zeros((rows, cols // 8), dtype=np.uint32)
for i in range(8):
    q |= w[:, i::8] << (4 * i)
```

Scales also get reordered (`_scale_perm` for groupsize=128, `_scale_perm_single`
for full-channel). Marlin further enforces `groupsize ∈ {-1, 128}` and
`bits == 4` in the upstream kernel.

The **same `[0,2,4,6,1,3,5,7]` interleave** appears in both AWQ and Marlin,
because both formats target the same `mma` lane assignment. The difference is
that AWQ applies it along the `n`-dim of the original weight layout while Marlin
applies it inside its tile-permuted layout.

### 4.6 GPTQModel vs AutoGPTQ

GPTQModel (ModelCloud fork) is wire-compatible with AutoGPTQ: same
`qweight/qzeros/scales/g_idx` keys, same dtypes and packing. It introduces
**format flags** in `quantize_config.json`:

```
format: GPTQ | GPTQ_V2 | MARLIN | BITBLAS
```

`GPTQ_V2` adds a **proper symmetric quant** (no +1 hack; zeros are simply
stored as 0 with `sym=True`). On disk a `GPTQ_V2` checkpoint and a `GPTQ`
checkpoint with the same model differ only by the absence of the +1 offset; you
need to read `quantize_config.json` to know which to apply.

GPTQModel also exports AWQ (`FORMAT.GEMM` / `FORMAT.GEMV`), GGUF, FP8, and EXL3.
Each format reuses its native upstream key naming.

---

## 5. AutoRound

Source: `intel/auto-round/auto_round/export/export_to_autoround/export.py`,
README.

### 5.1 What container

AutoRound exports to **five** formats from one quantized model:
`auto_round` (its own native), `auto_gptq`, `auto_awq`, `llm_compressor`,
`gguf`. The native `auto_round` format is essentially **AutoGPTQ's container
with extended metadata**:

- Same tensor naming (`qweight/qzeros/scales/g_idx`).
- Same INT32 packing along `k` for INT4.
- `quantization_config.json` adds:
  - `quant_method: "auto-round"` (transformers-routable string)
  - `packing_format: "auto_round:exllamav2"` or similar (`auto_round:gptqmodel`,
    `auto_round:auto_awq`, etc.)
  - `act_bits` (for W4A8/W4A4 schemes; AutoGPTQ has no equivalent)
  - `extra_config` (per-layer override dict — mixed precision recipes)
  - `block_name_to_quantize` (transformer block prefix, e.g. `model.layers`)

### 5.2 INT4 vs INT8 layout

INT4: identical to AutoGPTQ — `qweight (k_in/8, n_out) int32`, low→high nibbles,
+1 zero offset (asymmetric default).

INT8: there is no INT8 packing; INT8 weights are stored as `int8` directly with
per-group `scales` (and `zp` if `sym=False`). Shapes:

| key | shape | dtype |
|---|---|---|
| `*.weight` | `(out_features, in_features)` | `int8` |
| `*.weight_scale` | `(out_features, in_features // group_size)` | `float16` |
| `*.weight_zero_point` | same | `int8` (only if asym) |

### 5.3 Defaults

- Per-group with `group_size=128`. Per-channel is achieved by `group_size=-1`
  (or `group_size=in_features`).
- **`sym=False`** by default for W4A16 (asymmetric, matches AutoGPTQ default).
- W4A16 = "int4 weight, fp16 activation". Other schemes: W4A8, W2A16G64 etc.
- `iters=200`, `nsamples=128` for the calibration loop.

### 5.4 Unique fields

`enable_alg_ext` (turn on the SignRound-V2 / GPTQ+AR variant), `disable_opt_rtn`
(skip rounding optimisation, useful for iters=0 plain RTN export), `act_scale`
+ `w_bf16_to_fp8_scale` for the FP8 export path.

---

## 6. Comparison table

For Qwen3-35B (~35.0 B params, attn+mlp largely fp16; using ~33.5 B
quantizable params after embed/lm-head/norms), file-size estimates assume the
**embeddings stay at FP16** and the LM head at Q6_K/FP16, which is what
llama.cpp does for the `_K_M` recipes. Numbers cross-checked against published
Qwen3-class 35B GGUFs (Unsloth / bartowski).

| Format | bits/w | group | sym/asym | act_order | zeros stored | dequant cost | Qwen3-35B size |
|---|---|---|---|---|---|---|---|
| FP16 | 16 | – | – | – | – | nil | ~70 GB |
| Q8_0 | 8.5 | 32 | sym | n/a | implicit (sym) | 1 fp16 mul | ~37 GB |
| Q6_K | 6.5625 | 16+256 | sym (offset 32) | n/a | implicit | 2 mul + bit ops | ~29 GB |
| Q5_K_M | 5.5 | 32+256 | asym | n/a | per-sub-block min | 2 mul + 1 sub | ~26 GB |
| Q4_K_M | 4.5 | 32+256 | asym | n/a | per-sub-block min | 2 mul + 1 sub | ~22 GB |
| AWQ INT4 g128 | 4.25 | 128 | asym | no | per-group, 4-bit packed | 1 mul + 1 sub | ~19 GB |
| GPTQ INT4 g128 | 4.25 | 128 | asym | optional | per-group, 4-bit packed (+1) | 1 mul + 1 sub | ~19 GB |
| GPTQ INT4 g128 sym (V2) | 4.25 | 128 | sym | optional | none | 1 mul | ~19 GB |
| AutoRound INT4 g128 | 4.25 | 128 | asym (default) | no | 4-bit packed | same as GPTQ | ~19 GB |

bits/w for AWQ/GPTQ at g128 = 4 + 16/128 (scale fp16) + 4/128 (qzero) ≈ 4.156,
plus packing overhead ~4.25. K-quant numbers are exact super-block formulas.
Q4_K_M includes Q6_K for `output.weight`/`*.attn_v` and Q4_K elsewhere — that's
the "_M" tail.

---

## 7. Dequant kernel guidance for Xe2

**Q4_K / Q5_K**: One super-block (256 weights, 144 / 176 bytes) fits in two
128-bit cache lines plus a tail. Plan: one work-item per **64-element half
super-block** (i.e. the two-sub-block unit that shares `q[is..is+1]`).
Pre-decode the 12-byte `scales[]` into 8 `half` scales and 8 `half` mins in SLM
once per super-block, then each work-item reads 32 packed bytes of `qs`
(plus 32 of `qh` for Q5_K), unpacks low/high nibbles in registers, and
multiply-subtracts in fp16. The 12-byte scale field is 96 bits — load as
`uint3` and bit-extract using the `get_scale_min_k4` recurrence; do **not**
loop, unroll all 8 lanes. Beware: low and high nibbles of `qs[l]` belong to
**different** sub-blocks; do the two halves as separate fma chains, not
interleaved.

**Q6_K**: Each 128-element half-block reads `ql[64] + qh[32] + scales[8] +
d`. The natural Xe2 layout is one work-item per 32 outputs (the `l ∈ [0,32)`
inner loop), with `scales[8]` pre-broadcast in registers and `qh` byte
broadcast across four lanes. Beware of sign: the cast `(int8_t)((ql|qh<<4))-32`
must be done at int8, not uint8, otherwise the `-32` lands in the wrong domain.

**Q8_0**: trivial — fuse with the matmul. No need to dequantize separately;
emit `int8 → fp16` conversion in the inner-loop and multiply by the per-block
`d` once every 32 K-steps.

**AWQ INT4**: `qweight` is row-major `(k, n/8) int32`. For a typical
`mma.m16n8k16`-class op on Xe2 (DPAS m=8 n=8 k=16 INT8/INT4), a thread that
owns 8 contiguous output columns of one row reads exactly **one int32**, applies
shift-by-`4*AWQ_REVERSE_ORDER[lane]`, mask 0xF, subtracts the 8 unpacked qzero
nibbles (loaded once per group), multiplies by the 8 fp16 scales (also once per
group). Keep the AWQ permutation in a constant register; do not recompute. The
fact that AWQ packs along `n` is what makes the per-row dequant a single int32
load — this is the entire point of the AWQ_ORDER trick.

**GPTQ INT4 (no act_order)**: `qweight` is `(k/8, n) int32`, packed along **k**.
That means a thread owning 8 contiguous K elements of one column reads one
int32, shifts by `4*(k%8)` (no permutation), masks, subtracts `(z_pkg+1)`,
multiplies by `s`. Don't forget the +1 — encode it as a constant baked into the
zero-point register. For Marlin layout, you need the tile permutation done at
load time; my plan is to do AutoGPTQ-style on-disk and Marlin-repack into a
device-side scratch buffer at model load (one-shot, not per-forward).

**GPTQ INT4 with act_order**: pre-permute weights at load via `g_idx_inv`,
store as if `desc_act=False`, and apply `g_idx` to the **activation** vector
each forward. On Xe2 the activation permutation is a single SLM scatter — cheap
once per token, not per K-step.

**Alignment hazards**: GGUF tensor offsets are aligned to 32 bytes; `block_q4_K`
is 144 B (multiple of 16, but not 32). Per-tensor base is aligned, but the *N*-th
block inside a tensor lands at offset `144*N` from that base — fine for byte
loads but not safe for 128-bit aligned vector loads. AWQ/GPTQ qweight is int32
arrays with safetensors' default 8-byte alignment; OK for `int4` vector loads.

---

## 8. Calibration / conversion commands for Qwen3-35B

Assume `~/models/Qwen3-35B-fp16/` is a HuggingFace checkpoint. Calibration set:
WikiText-2 / pile-128, 128 samples × 2048 tokens (defaults work).

### 8.1 GGUF Q4_K_M, Q5_K_M, Q6_K, Q8_0 (llama.cpp)

```bash
# 1. HF → GGUF FP16 (one-shot, ~70 GB output)
python convert_hf_to_gguf.py ~/models/Qwen3-35B-fp16 \
    --outfile /tmp/qwen3-35b-f16.gguf --outtype f16

# 2. quantize variants
./build/bin/llama-quantize /tmp/qwen3-35b-f16.gguf qwen3-35b-Q4_K_M.gguf Q4_K_M  # ~22 GB
./build/bin/llama-quantize /tmp/qwen3-35b-f16.gguf qwen3-35b-Q5_K_M.gguf Q5_K_M  # ~26 GB
./build/bin/llama-quantize /tmp/qwen3-35b-f16.gguf qwen3-35b-Q6_K.gguf   Q6_K    # ~29 GB
./build/bin/llama-quantize /tmp/qwen3-35b-f16.gguf qwen3-35b-Q8_0.gguf   Q8_0    # ~37 GB
```

Optional importance-matrix pass (`llama-imatrix --in-file calib.txt -m
qwen3-35b-f16.gguf -o imat.dat`) followed by `llama-quantize --imatrix imat.dat
... Q4_K_M` — saves 1–2 GB while improving perplexity.

### 8.2 AWQ INT4 g128 (AutoAWQ)

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

m = AutoAWQForCausalLM.from_pretrained("~/models/Qwen3-35B-fp16",
                                       device_map="auto", safetensors=True)
t = AutoTokenizer.from_pretrained("~/models/Qwen3-35B-fp16")
m.quantize(t, quant_config={
    "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"
})
m.save_quantized("./qwen3-35b-awq-int4-g128")     # ~19 GB
t.save_pretrained("./qwen3-35b-awq-int4-g128")
```

### 8.3 GPTQ INT4 g128 (AutoGPTQ / GPTQModel)

```python
from gptqmodel import GPTQModel, QuantizeConfig

cfg = QuantizeConfig(bits=4, group_size=128, desc_act=False, sym=False)
m = GPTQModel.load("~/models/Qwen3-35B-fp16", cfg)
m.quantize(calibration_dataset)   # list[str] or list[dict]
m.save("./qwen3-35b-gptq-int4-g128")               # ~19 GB
```

For Marlin-ready output, set `format=FORMAT.MARLIN` in `QuantizeConfig`. For the
GPTQ-V2 (no +1 hack) variant, set `sym=True, format=FORMAT.GPTQ_V2`.

### 8.4 AutoRound INT4

```bash
auto-round \
    --model ~/models/Qwen3-35B-fp16 \
    --scheme W4A16 \
    --format auto_round,auto_gptq,auto_awq \
    --group_size 128 \
    --iters 200 --nsamples 128 \
    --output_dir ./qwen3-35b-autoround-int4   # ~19 GB per format
```

`--format` accepts a comma-separated list and produces one folder per container.
For a GGUF Q4_K_M handoff: `--scheme GGUF:Q4_K_M --format gguf --iters 0` (RTN
mode; AutoRound's GPTQ-style optimisation does not apply to GGUF k-quants).

### 8.5 Estimated file sizes (Qwen3-35B)

| Output | Estimated size |
|---|---|
| FP16 GGUF | ~70 GB |
| Q8_0 | ~37 GB |
| Q6_K | ~29 GB |
| Q5_K_M | ~26 GB |
| Q4_K_M | ~22 GB |
| AWQ INT4 g128 | ~19 GB |
| GPTQ INT4 g128 | ~19 GB |
| AutoRound INT4 g128 (auto_round / auto_gptq / auto_awq containers) | ~19 GB each |

---

## Ambiguities I could not resolve

1. **AWQ + Marlin repack permutation.** AutoAWQ's `WQLinear_Marlin` exists, but
   I could not nail down whether it shares the exact `_perm` and `_scale_perm`
   tables from `IST-DASLab/marlin`, or a slightly different variant tuned for
   AWQ's pre-permuted weight tensor. Likely identical, but I would dump both
   and `assert torch.equal` before trusting it.
2. **GPTQ_V2 disk format vs vanilla GPTQ.** The README says "no +1 offset" but
   I did not find a definitive test vector. A V2 checkpoint produced by
   GPTQModel and a V1 checkpoint of the same weights should differ only in the
   `qzeros` tensor (and maybe a flag in `quantize_config.json`). I want to
   round-trip a tiny model (e.g. Qwen3-0.6B) through both and diff the
   tensors before writing the loader branch.
3. **AutoRound native `packing_format` strings.** I saw at least four values
   (`auto_round:exllamav2`, `auto_round:gptqmodel`, `auto_round:auto_awq`,
   `auto_round:tritonv2`) but the canonical list is not in the README; the
   safe path is to match on prefix `auto_round:` and route the suffix to the
   matching loader.
4. **Marlin `_scale_perm` formula.** The IST-DASLab `__init__.py` constructs
   it inline but I did not transcribe the exact math; for a hand-written Xe2
   port, a one-time Python dump (`print(_scale_perm.tolist())`) is more
   reliable than reverse-engineering.
5. **GGUF MXFP4 (type 39) layout.** New in early 2026, not yet documented in
   `gguf.md`. Spec lives in PR commits only as of this writing — I am
   deferring it until I actually see an MXFP4 model in the wild.
6. **Q8_K (type 15)** is used as an internal "activation" format inside
   llama.cpp k-quant matmuls and I did not verify whether it is ever written
   to disk. Treat as in-memory only unless proven otherwise.

---

### Sources

- ggml GGUF spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- ggml-common.h (k-quant structs): https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h
- ggml-quants.c (dequant reference): https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c
- ggml.h (`ggml_type` enum): https://github.com/ggml-org/ggml/blob/master/include/ggml.h
- AutoAWQ packing utils: https://github.com/casper-hansen/AutoAWQ/blob/main/awq/utils/packing_utils.py
- AutoAWQ WQLinear_GEMM: https://github.com/casper-hansen/AutoAWQ/blob/main/awq/modules/linear/gemm.py
- AutoGPTQ qlinear_cuda: https://github.com/AutoGPTQ/AutoGPTQ/blob/main/auto_gptq/nn_modules/qlinear/qlinear_cuda.py
- GPTQModel: https://github.com/ModelCloud/GPTQModel
- IST-DASLab Marlin: https://github.com/IST-DASLab/marlin/blob/master/marlin/__init__.py
- vLLM gptq_marlin kernel: https://github.com/vllm-project/vllm/tree/main/csrc/quantization/gptq_marlin
- intel/auto-round: https://github.com/intel/auto-round
- AutoRound AWQ format wiki: https://deepwiki.com/intel/auto-round/9.3-awq-format
- Qwen3-35B GGUF size reference: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF
