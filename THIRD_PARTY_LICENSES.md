# Third-party notices & IP provenance

This engine is original work. The performance-critical core — the SYCL/GPU
kernels, the DeltaNet/gated-attention/MoE forward passes, the model
orchestration, the server, and the CLI — is implemented from scratch against
the **published model architecture specifications** (Qwen3.6 `config.json`,
the DeltaNet / gated-attention research papers), not copied from any other
inference engine.

This file records the few places where the engine (a) vendors a third-party
library, (b) is **format-compatible** with an external file format, or (c)
contains code **derived** from an external MIT-licensed source. All such
sources are permissively licensed (MIT) and compatible with this project's
intended Apache-2.0 open-core release.

---

## A. Vendored libraries (`third_party/`)

| Component | Version | License | Use |
|---|---|---|---|
| [cpp-httplib](https://github.com/yhirose/cpp-httplib) | v0.18.7 | MIT | `ie serve` HTTP layer |
| [nlohmann/json](https://github.com/nlohmann/json) | v3.11.3 | MIT | JSON parsing (server, HF-import config) |

Both are unmodified header-only drops. Their copyright/permission notices are
retained in `third_party/httplib/httplib.h` and `third_party/nlohmann/json.hpp`.

## B. Format compatibility (functional, not derivative)

The engine reads and writes the **GGUF** container format and decodes the
**Q4_K / Q5_K / Q6_K / Q8_0** quantization block layouts so it can load the
same model files the ecosystem ships. A file format is a functional
specification: matching its byte layout to interoperate is not copying its
implementation. The following reference the GGUF/ggml *format* (key names,
`ggml_type` enum integer values, `ne[]` dimension ordering) as data, and
contain original reader/writer code:

- `src/loaders/gguf_reader.cpp`, `include/ie/gguf_writer.hpp`
- `include/ie/dtype.hpp` (enum **values** match `ggml_type` so on-disk type
  IDs round-trip — these are facts, not authored expression)
- `src/tokenizer/tokenizer.cpp` (reads `tokenizer.ggml.*` metadata keys)

## C. Derived from ggml (MIT) — attribution **required**

The **model-import path only** (`ie import`, used to ingest AWQ/GPTQ
safetensors) contains code that is a faithful port of routines in
`ggml-quants.c`. These are genuine derivative works of MIT-licensed ggml and
are retained under, and require inclusion of, the ggml MIT notice reproduced
in §D:

| File | Derived from (`ggml-quants.c`) |
|---|---|
| `src/loaders/quantize_q4k.cpp` | `quantize_row_q4_K_ref`, `make_qkx2_quants`, `get_scale_min_k4` |
| `src/loaders/quantize_q6k.cpp` | `quantize_row_q6_K_ref`, `make_qx_quants` |
| `include/ie/dequant_ref.hpp` | reference dequant (host-side validation golden) |

**Scope note:** none of these run in the inference hot path. The shipped
decode/prefill kernels (`src/ops/*`, `src/model/*`) are original. The ported
encoders exist so `ie import` can re-quantize dequantized AWQ/GPTQ weights
into GGUF blocks bit-faithfully; matching ggml's exact rounding is the whole
point (the output must be a valid ggml block), which is why these are ports
rather than independent rewrites.

> If a fully clean-room import path is ever desired, the Qx_K **encoders**
> could be re-derived from the block-format spec alone (the layout is
> functional). Given ggml's MIT terms this is optional, not required — the
> attribution in this file already satisfies the license.

## D. Developer-only tooling (not distributed with the engine)

`tools/llama_dump.cpp` is a validation harness that **links against a separate
llama.cpp checkout** to dump reference activations for bit-exact parity
testing. It is **not** part of any default build target (see
`tools/CMakeLists.txt`) and is **not** shipped in the product. It is an
external measuring stick, exactly like running `llama-perplexity` as an oracle
— it puts no llama.cpp code into the engine binary. Public releases should
exclude it (or gate it behind an opt-in dev flag) to keep the distributed tree
free of ggml-linked code.

---

## ggml / llama.cpp MIT license (covers §C)

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

cpp-httplib: Copyright (c) 2025 Yuji Hirose — MIT.
nlohmann/json: Copyright (c) 2013-2023 Niels Lohmann — MIT.

*Maintained as part of the launch-readiness (P4) checklist. Update when a new
third-party or derived component is added.*
