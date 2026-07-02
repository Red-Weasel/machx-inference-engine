# Intel Arc Pro B70 - Hardware Brief for a Native LLM Inference Engine

*Compiled 2026-04-25. Every numeric claim is cited inline. All numbers below are vendor / press / Khronos sources unless explicitly flagged "extrapolated."*

---

## 1. Existence Check

**Yes - the Arc Pro B70 is a publicly announced and shipping product as of 2026-04-25.** Intel launched it on **March 25, 2026** at **$949** alongside the smaller Arc Pro B65 ([Phoronix - "Intel Announces Arc Pro B70 With 32GB GDDR6"](https://www.phoronix.com/news/Intel-Arc-Pro-B70-Announced); [VideoCardz launch news](https://videocardz.com/newz/intel-launches-arc-pro-b70-at-949-with-32gb-gddr6-memory)). The product page is live on intel.com ([ARK SKU 245797](https://www.intel.com/content/www/us/en/products/sku/245797/intel-arc-pro-b70-graphics/specifications.html)) and an [official datasheet PDF](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/2026-03/datasheet-b70-gpu.pdf) was published in March 2026. Phoronix has done a [full Linux LLM/AI/OpenCL/Vulkan review](https://www.phoronix.com/review/intel-arc-pro-b70-linux). Intel's own Newegg / Micro Center / B&H listings show retail availability ([Newegg N82E16814883008](https://www.newegg.com/intel-arc-pro-b70-32gb-graphics-card/p/N82E16814883008)).

The B70 is the workstation-Pro flavor of the **BMG-G31 "Big Battlemage"** die, the same silicon Intel is expected to ship in a consumer Arc B770 SKU later in 2026 ([TechSpot - Big Battlemage at CES 2026](https://www.techspot.com/news/110537-intel-confirms-big-battlemage-gpu-arc-b770-could.html)).

---

## 2. Spec Sheet

| Item | Value | Source |
|------|-------|--------|
| Architecture | Xe2-HPG ("Battlemage"), die BMG-G31 | [GPUPoet B70 page](https://gpupoet.com/gpu/learn/card/intel-arc-pro-b70), [PMZFX hardware.md](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks/blob/master/hardware.md) |
| Process node | TSMC N5 (5 nm) | [Tom's Hardware launch](https://www.tomshardware.com/pc-components/gpus/intel-arc-pro-b70-and-arc-pro-b65-gpus-bring-32gb-of-ram-to-ai-and-pro-apps-bigger-battlemage-finally-arrives-but-its-not-for-gaming), [CraftRigs review](https://craftrigs.com/articles/intel-arc-pro-b70-specs-27b-models/) |
| Xe2 cores | **32** (full BMG-G31) | [Igor's Lab launch](https://www.igorslab.de/en/intel-launches-the-arc-pro-b70-with-32-gb-of-vram-for-949-big-battlemage-is-coming-first-for-ai-and-workstations/), [PMZFX](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks/blob/master/hardware.md) |
| XVE (vector engines) | **256** (8 per Xe-core × 32 cores) | derived from [HWCooling Xe2 analysis](https://www.hwcooling.net/en/batttlemage-details-of-intel-xe2-gpu-architecture-analysis/) (8 XVE/Xe-core) |
| XMX engines | **256** (8 per Xe-core × 32) | [Igor's Lab](https://www.igorslab.de/en/intel-launches-the-arc-pro-b70-with-32-gb-of-vram-for-949-big-battlemage-is-coming-first-for-ai-and-workstations/), [ASRock B70 Creator listing](https://www.asrock.com/Graphics-Card/Intel/Intel%20Arc%20Pro%20B70%20Creator%2032GB/) |
| Ray-tracing units | 32 | [PMZFX](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks/blob/master/hardware.md) |
| Boost / max dynamic clock | **2800 MHz** (typical sustained ~2280-2540 MHz) | [Puget B70 review](https://www.pugetsystems.com/labs/articles/intel-arc-pro-b70-review/), [ASRock B70 Creator (2540 MHz "GPU clock")](https://www.asrock.com/Graphics-Card/Intel/Intel%20Arc%20Pro%20B70%20Creator%2032GB/) |
| Peak FP32 | **22.94 TFLOPS** | [GPUPoet](https://gpupoet.com/gpu/learn/card/intel-arc-pro-b70), [Puget](https://www.pugetsystems.com/labs/articles/intel-arc-pro-b70-review/) |
| Peak FP16 (matrix, XMX, dense) | **45.88 TFLOPS** vector path; XMX dense FP16 = **183 TFLOPS** (derived: 367 INT8 TOPS / 2) | FP16 vector: [GPUPoet](https://gpupoet.com/gpu/learn/card/intel-arc-pro-b70); Xe2 ratio (INT8 = 2 × FP16) confirmed by [Intel VTune XMX docs](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2025-1/xve-fp16-bf16-int8-int4-int2-xmx-instructions.html) (FP16=2048 ops/clk, INT8=4096 ops/clk per Xe core) |
| Peak BF16 (XMX) | ≈ FP16 (Xe2 BF16 throughput is identical to FP16) | [VTune XVE/XMX docs](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2025-1/xve-fp16-bf16-int8-int4-int2-xmx-instructions.html), [HWCooling](https://www.hwcooling.net/en/batttlemage-details-of-intel-xe2-gpu-architecture-analysis/) |
| Peak INT8 (XMX, dense) | **367 TOPS** | [Intel ARK B70](https://www.intel.com/content/www/us/en/products/sku/245797/intel-arc-pro-b70-graphics/specifications.html) (via search), [TheFPSReview](https://www.thefpsreview.com/2026/03/25/intels-big-battlemage-finally-arrives-arc-pro-b70-and-b65-launched-today-with-32gb-of-vram-and-up-to-367-tops/) |
| Peak INT4 (XMX, dense) | **≈ 734 TOPS** (extrapolated: Xe2 INT4 = 2 × INT8 throughput) | derived from [VTune docs](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2025-1/xve-fp16-bf16-int8-int4-int2-xmx-instructions.html) and [HWCooling Xe2 analysis](https://www.hwcooling.net/en/batttlemage-details-of-intel-xe2-gpu-architecture-analysis/) |
| Peak INT2 | ≈ 1468 TOPS (extrapolated) | [HWCooling](https://www.hwcooling.net/en/batttlemage-details-of-intel-xe2-gpu-architecture-analysis/) |
| VRAM capacity | **32 GB GDDR6** (with ECC) | [PMZFX](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks/blob/master/hardware.md), [GPUPoet](https://gpupoet.com/gpu/learn/card/intel-arc-pro-b70) |
| Memory speed | 19 Gbps | [ASRock B70 Creator](https://www.asrock.com/Graphics-Card/Intel/Intel%20Arc%20Pro%20B70%20Creator%2032GB/) |
| Memory bus | **256-bit** | [ASRock](https://www.asrock.com/Graphics-Card/Intel/Intel%20Arc%20Pro%20B70%20Creator%2032GB/), [GPUPoet](https://gpupoet.com/gpu/learn/card/intel-arc-pro-b70) |
| Peak memory bandwidth | **608 GB/s** | [Phoronix](https://www.phoronix.com/news/Intel-Arc-Pro-B70-Announced), [GPUPoet](https://gpupoet.com/gpu/learn/card/intel-arc-pro-b70) |
| L2 cache | **18 MB** (whole-chip; same shared cache architecture as B580 BMG-G21) | [ChipsAndCheese - Battlemage architecture](https://chipsandcheese.com/p/intels-battlemage-architecture); BMG-G21 also uses 18 MB per [HWCooling](https://www.hwcooling.net/en/batttlemage-details-of-intel-xe2-gpu-architecture-analysis/). **Not yet independently confirmed for BMG-G31 specifically.** |
| L1 cache / SLM (per Xe-core) | **256 KB unified** (L1+SLM split, ChipsAndCheese measured) | [ChipsAndCheese](https://chipsandcheese.com/p/intels-battlemage-architecture) |
| L1 cache effective | 96 KB (with full SLM allocation), up to 256 B/cycle | [ChipsAndCheese](https://chipsandcheese.com/p/intels-battlemage-architecture) |
| Register file per XVE | **64 KB** (up to 8 threads × 8 KB) | [ChipsAndCheese](https://chipsandcheese.com/p/intels-battlemage-architecture) |
| PCIe | **PCIe 5.0 x16** physical (some AIB/dense variants negotiate x8) | [ASRock](https://www.asrock.com/Graphics-Card/Intel/Intel%20Arc%20Pro%20B70%20Creator%2032GB/), [Phoronix](https://www.phoronix.com/news/Intel-Arc-Pro-B70-Announced), [PMZFX](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks/blob/master/hardware.md) |
| TBP (Intel reference) | **230 W** | [Phoronix](https://www.phoronix.com/news/Intel-Arc-Pro-B70-Announced), [Puget](https://www.pugetsystems.com/labs/articles/intel-arc-pro-b70-review/) |
| TBP (AIB partner range) | 160 W - 290 W | [Phoronix](https://www.phoronix.com/news/Intel-Arc-Pro-B70-Announced) |
| Power connector | 1× 12V-2x6 | [ASRock](https://www.asrock.com/Graphics-Card/Intel/Intel%20Arc%20Pro%20B70%20Creator%2032GB/) |
| MSRP (Intel reference) | **$949** | [Phoronix](https://www.phoronix.com/news/Intel-Arc-Pro-B70-Announced), [Igor's Lab](https://www.igorslab.de/en/intel-launches-the-arc-pro-b70-with-32-gb-of-vram-for-949-big-battlemage-is-coming-first-for-ai-and-workstations/) |
| Display outputs | 4× DisplayPort 2.1 (1× UHBR13.5, 3× UHBR10) | [ASRock](https://www.asrock.com/Graphics-Card/Intel/Intel%20Arc%20Pro%20B70%20Creator%2032GB/), [Igor's Lab](https://www.igorslab.de/en/intel-launches-the-arc-pro-b70-with-32-gb-of-vram-for-949-big-battlemage-is-coming-first-for-ai-and-workstations/) |
| PCI ID | `8086:e223` | [PMZFX](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks/blob/master/hardware.md) |

### Quick LLM-relevant figures

- **Compute:bandwidth ratio (FP16 dense XMX):** ≈183 TFLOPS / 608 GB/s = **301 FLOP / B**. FP16 GEMV is therefore strongly bandwidth-bound (LLM decode); for KV-cache-heavy steps, design around 608 GB/s.
- **VRAM / parameter capacity:** 32 GB easily fits a 30B-parameter Q4 model (~17-18 GB), Q8 14B (~14 GB), or two 7B models pinned simultaneously.

---

## 3. Architecture Details for Kernel Writers

### 3.1 SIMD / subgroup width

Xe2 dropped Alchemist's SIMD8 mode and **runs natively at SIMD16, with SIMD32 also supported** ([ChipsAndCheese - Battlemage](https://chipsandcheese.com/p/intels-battlemage-architecture); [HWCooling](https://www.hwcooling.net/en/batttlemage-details-of-intel-xe2-gpu-architecture-analysis/)). Subgroup-size 8 is still legally exposed by `cl_intel_subgroup_matrix_multiply_accumulate` for legacy reasons but **prefer subgroup 16 on Battlemage** - it is the native execution width and unlocks the FP16/BF16 same-precision accumulator path ([Khronos extension spec](https://github.com/khronosgroup/OpenCL-Docs/blob/main/extensions/cl_intel_subgroup_matrix_multiply_accumulate.asciidoc)). Each Xe-core has 8 XVEs, so 32 cores × 8 XVE × SIMD16 = nominally 4096 lanes in flight per cycle.

### 3.2 XMX matrix tile shapes (`cl_intel_subgroup_matrix_multiply_accumulate`)

Pulled from the [Khronos extension spec](https://github.com/khronosgroup/OpenCL-Docs/blob/main/extensions/cl_intel_subgroup_matrix_multiply_accumulate.asciidoc); these are the legal `(M, K)` for each dtype, with N pinned to subgroup size (8 or 16):

| dtype | Subgroup | M values | K | Acc / Result type |
|------|----------|----------|---|-------------------|
| FP16 | 8 or **16** | 1, 2, 4, **8** | **16** | FP32 (or FP16 acc if SG=16) |
| BF16 | 8 or **16** | 1, 2, 4, **8** | **16** | FP32 (or BF16 acc if SG=16) |
| INT8 | 8 or **16** | 1, 2, 4, **8** | **32** | INT32 |
| INT4 | 8 or **16** | 1, 2, 4, **8** | **64** | INT32 |

The standard "fat" tile to write your inner-loop microkernel against is therefore:

- **FP16/BF16:** `M=8, N=16, K=16` -> 8×16×16 = 2048 MACs per `dpas` per subgroup
- **INT8:** `M=8, N=16, K=32` -> 4096 MACs per `dpas`
- **INT4:** `M=8, N=16, K=64` -> 8192 MACs per `dpas`

Doubling-from-FP16 ratios (FP16 -> BF16 = 1×, INT8 = 2×, INT4 = 4×, INT2 = 8×) per Xe-core per clock are confirmed by [Intel's VTune XMX-instructions reference](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2025-1/xve-fp16-bf16-int8-int4-int2-xmx-instructions.html).

### 3.3 Mixed-precision: INT4 weight × FP16 activation

**Direct mixed-dtype `dpas` is not exposed** by the Khronos extension - both A and B must share dtype. For the LLM weight-only-quant path (W4A16, W4A8, W8A16) you have two production-tested options:

1. **Dequant-fused-into-load**: dequantize the INT4 weights to FP16/BF16 in registers right before feeding XMX. This is what Intel's own [`sycl-tla` SYCL CUTLASS port](https://github.com/intel/sycl-tla/blob/main/README.md) does - the README explicitly lists "mixed dtype GEMM support including FP16 + INT8 and FP16 + INT4 with tensor-wise, channel-wise, and group-wise quantization."
2. **Quantize activations to INT8 before the matmul** and use the INT8 XMX path - cheaper in compute but with the usual W4A8 accuracy hit.

There is no `dpas`-level support for FP8 (E4M3 / E5M2) on Xe2 ([SYCL-TLA README](https://github.com/intel/sycl-tla/blob/main/README.md) lists FP8 only for forward-compat / Xe3); you can software-emulate via FP16 accumulation but you don't get the throughput multiplier.

### 3.4 SLM, register file, 2D block load

- **SLM per work-group:** 128 KB hard cap on Xe-HPG/Xe2 (the 256 KB physical block is shared between L1 and SLM; SLM allocations >128 KB starve L1). [ChipsAndCheese measured ~15 ns SLM latency](https://chipsandcheese.com/p/intels-battlemage-architecture).
- **Register file per XVE:** 64 KB. Up to 8 hardware threads × 8 KB regs each, or fewer threads × bigger registers (e.g. 4 threads × 16 KB) for higher per-thread occupancy. ([ChipsAndCheese](https://chipsandcheese.com/p/intels-battlemage-architecture))
- **2D block load / store:** Battlemage exposes the `intel_subgroup_block_read_2d_*` family (the same intrinsics the Xe-HPC PVC stack added). The IGC and compute-runtime pipeline has [explicit handling for 2D-block prefetch cache controls](https://dgpu-docs.intel.com/releases/rolling-release-notes.html). These are how you stream A/B tiles into registers in one transaction without going through SLM - **use them, do not build your own gather-scatter loaders.**
- **Async copy equivalent:** there is no NVIDIA-style `cp.async` on Xe2. The SLM path is synchronous; pipeline overlap relies on the compiler scheduling 2D-block-loads against `dpas` and the 8-way thread-level parallelism per XVE. See `sycl-tla`'s pipeline implementations for the canonical pattern.

### 3.5 DPAS instruction

The matrix instruction is **DPAS (Dot-Product Accumulate Systolic)**, present since Xe-HPG. ([Intel - "Improve AI Efficiency with Intel XMX"](https://www.intel.com/content/www/us/en/developer/articles/technical/use-new-built-in-ai-acceleration-engines.html)). Each Xe2 Xe-core has 8 × 2048-bit XMX engines, giving 2048 FP16-MACs / clock / Xe-core ([HWCooling](https://www.hwcooling.net/en/batttlemage-details-of-intel-xe2-gpu-architecture-analysis/)). At 2.8 GHz × 32 cores × 2048 MAC × 2 ops/MAC = **376 TFLOPS FP16**, which matches the 367 INT8-TOPS spec at slightly lower sustained clocks.

---

## 4. Comparison Table

| GPU | Peak FP16 (matrix) | Peak INT8 (matrix) | VRAM | Bandwidth | TBP | MSRP |
|-----|-------------------:|-------------------:|-----:|----------:|----:|-----:|
| **Arc Pro B70** (BMG-G31, 32 Xe cores) | ~183 TFLOPS¹ | **367 TOPS** | 32 GB GDDR6 | **608 GB/s** | 230 W | $949 |
| **Arc Pro B60** (BMG-G21, 20 Xe cores) | ~98 TFLOPS¹ | **197 TOPS** | 24 GB GDDR6 | 456 GB/s | 200 W | ~$500 |
| **Arc B580** (BMG-G21, 20 Xe cores, consumer) | ~117 TFLOPS² | **233 TOPS** | 12 GB GDDR6 | 456 GB/s | 190 W | $249 |
| **Arc A770** (Xe-HPG, 32 Xe cores) | **157 TFLOPS** | **262 TOPS** | 16 GB GDDR6 | 560 GB/s | 225 W | $349 (launch) |
| **RTX 4070 Ti SUPER** (AD103) | **88.2 TFLOPS** dense / 176.4 sparse | **353 TOPS** dense / 706 sparse | 16 GB GDDR6X | **672 GB/s** | 285 W | $799 |
| **RX 7700 XT** (Navi 32) | **70.3 TFLOPS** matrix | **70.3 TOPS** | 12 GB GDDR6 | 432 GB/s | 245 W | $449 |

Sources: B70 [Phoronix](https://www.phoronix.com/news/Intel-Arc-Pro-B70-Announced) / [GPUPoet](https://gpupoet.com/gpu/learn/card/intel-arc-pro-b70); B60 [Intel ARK 243916](https://www.intel.com/content/www/us/en/products/sku/243916/intel-arc-pro-b60-graphics/specifications.html); B580 [Intel ARK 241598](https://www.intel.com/content/www/us/en/products/sku/241598/intel-arc-b580-graphics/specifications.html); A770 [Intel ARK 229151](https://www.intel.com/content/www/us/en/products/sku/229151/intel-arc-a770-graphics-16gb/specifications.html); 4070 Ti SUPER [GPUPoet](https://gpupoet.com/gpu/learn/card/nvidia-geforce-rtx-4070-ti-super) / [WareDB](https://www.waredb.com/processor/nvidia-geforce-rtx-4070-ti-super); 7700 XT [GPUPoet](https://gpupoet.com/gpu/learn/card/amd-radeon-rx-7700-xt) / [AMD product page](https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7700-xt.html).

¹ Derived as INT8/2 - Intel does not publish the matrix-FP16 TFLOPS for B70/B60 directly; only INT8 TOPS and FP16-vector TFLOPS appear on ARK.
² B580 FP16-vector path is 27.34 TFLOPS per [Intel ARK](https://www.intel.com/content/www/us/en/products/sku/241598/intel-arc-b580-graphics/specifications.html); matrix FP16 is INT8/2 ≈ 117.

**Key takeaway:** The B70's selling point is **VRAM × bandwidth at the price** ($30/GB and 608 GB/s for $949) - not raw matrix throughput. For dense compute the 4070 Ti SUPER beats it on INT8 dense, but the B70 has 2× the VRAM, which matters more for >13B-param LLMs.

---

## 5. Software Stack Reality Check (as of 2026-04-25)

### Compute-runtime / NEO

The current production stack is **`intel/compute-runtime` 26.09.37435.1** (released March 2026), which lists Battlemage at "Production" quality with OpenCL 3.0 and Level Zero 1.14. It bundles **IGC v2.30.1** and was validated on Ubuntu 24.04 LTS with kernel `6.14.0-35-generic` plus the Intel-specific `6.14.0-1011-intel` variant from the intel-graphics PPA ([Release page](https://github.com/intel/compute-runtime/releases/tag/26.09.37435.1)). BMG-G31 (the B70's die) is explicitly listed as a supported device in this and the immediately preceding 26.01 / 25.48 releases ([release notes index](https://github.com/intel/compute-runtime/releases)).

### Kernel

Battlemage requires **Linux 6.12+ for official Xe-driver support** (6.11 had experimental support behind `force_probe`). For Ubuntu, intel-graphics PPA shipping `6.14.0-100x-intel` is the actively-tested target for BMG ([Phoronix - Battlemage one year](https://www.phoronix.com/review/intel-b580-compute-one-year), [Ubuntu Battlemage thread](https://discourse.ubuntu.com/t/advanced-intel-battlemage-gpu-features-now-available-for-ubuntu-24-10/51616)).

### oneAPI

Latest is **Intel oneAPI 2025.3 / DPC++ 2025.3.x** (the Xe2 SYCL story is mature: "GPU dispatch now supports Battlemage architecture integrated and discrete graphics parts" - [oneAPI release notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-base-toolkit/2025.html)). The SYCL Joint Matrix extension explicitly enumerates Battlemage in its supported-device list, alongside Lunar Lake and PVC ([Intel - Programming Intel XMX with SYCL Joint Matrix](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/programming-intel-xmx-using-sycl-joint-matrix.html)). `oneAPI 2026.x` is not yet GA as of 2026-04-25; trunk DPC++ builds are tagged 2025.3.x in the llama.cpp issue tracker.

### Project Battlematrix / LLM-Scaler / vLLM container

Intel has shipped a Linux-first inference stack called **Project Battlematrix**: a containerized vLLM build with multi-GPU scaling, PCIe P2P, ECC/SRIOV and remote firmware ([Intel - Project Battlematrix overview](https://www.intel.com/content/www/us/en/developer/articles/technical/introduction-project-battlematrix.html), [Intel software update Aug 2025](https://www.intel.com/content/www/us/en/developer/articles/technical/battlematrix-software-update-august2025.html), [Tom's Hardware - LLM Scaler 1.0](https://www.tomshardware.com/pc-components/gpus/intel-releases-new-software-for-gpu-powered-project-battlematrix-workstations-arc-pro-b-series-gpus-get-llm-scaler-1-0-software-to-optimize-performance-in-ai-workloads)). This is the path Intel recommends for production inference; the container layer is what indexed the "B70" SKU before launch ([wccftech - LLM-Scaler vLLM release notes leak](https://wccftech.com/intel-confirms-arc-pro-b70-workstation-gpu-via-llm-scaler-vllm-ai-release-notes/)).

### IPEX-LLM

`intel/ipex-llm` officially supports the Arc B-series; the [Battlemage quickstart](https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Quickstart/bmg_quickstart.md) lists driver minima (Windows 32.0.101.6449+; Linux requires the intel-graphics PPA). Reported 8B Q4 throughput on B580 is ~62 t/s, with the B70 expected ~55-65 t/s on Llama-3-8B-Q4 and 18-22 t/s on Llama-3-70B-Q4 ([wccftech MLPerf 6.0 B70 numbers](https://wccftech.com/intel-arc-pro-b70-delivers-80-percent-boost-mlperf-inference-v6-0/)).

### llama.cpp

`-DGGML_SYCL=ON -DGGML_SYCL_DEVICE_ARCH=bmg_g31` builds work, but the SYCL backend is **markedly slower than Vulkan** on Battlemage today (Vulkan ~2× faster per [llama.cpp discussion #12570](https://github.com/ggml-org/llama.cpp/discussions/12570) and [Phoronix llama.cpp Vulkan EOY 2025](https://www.phoronix.com/review/llama-cpp-vulkan-eoy2025)). Mesa/ANV cooperative-matrix support for Xe2 has been merged ([Phoronix - Vulkan coop-matrix Xe2](https://www.phoronix.com/news/Intel-Xe2-Coop-Matrix-Enable)).

### VTune / unitrace

Intel **VTune Profiler** has BMG-G21 supported in the 2025.x releases and BMG-G31 was added in late 2025 / early 2026 release notes ([HotHardware - BMG-G31 VTune support](https://hothardware.com/news/intel-bmg-g31-vtune-profiler), [TechPowerUp coverage](https://www.techpowerup.com/343775/intel-arc-battlemage-bmg-g31-gpu-emerges-in-official-software-support)). `unitrace` (the ze_tracer / pti-gpu utility) works on Battlemage.

### ESIMD

The DPC++ ESIMD CM-style backend works on Xe2; the IGC vector backend lights up ESIMD on `bmg_g21`/`bmg_g31` targets in oneAPI 2025.3 ([Intel oneAPI 2025 release notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-base-toolkit/2025.html)).

---

## 6. Known Gotchas (mid-2026)

1. **SYCL Q8_0 weight-only matmul is bandwidth-starving on B70** - 21% of theoretical bandwidth (4.88 t/s) versus 53% for Q4_K_M (20.56 t/s) on Qwen3.5-27B because Q8_0 hits the slow DMMV path while Q4_K_M uses MMVQ+reorder. The fix (dequant-into-reorder) was filed as PR #21527 and gives ~3.1× speedup. See [llama.cpp issue #21517](https://github.com/ggml-org/llama.cpp/issues/21517). **If you're writing a Q8 inner kernel, do not assume you'll match Q4_K's bandwidth utilization out of the box.**

2. **Silent weight corruption with default SYCL reorder optimizations on B70** - the optimized paths produce hallucinated training-data fragments unless `GGML_SYCL_DISABLE_OPT=1` is set. Reproduced on Gemma-4-26B-A4B-Q6_K_XL with oneAPI 2025.3.2, level-zero 1.14.37435+1, llama.cpp commit e21cdc1. Symptom looks like a DPAS sync / tensor-reorder bug in the Xe2-specific kernel. [llama.cpp issue #21893](https://github.com/ggml-org/llama.cpp/issues/21893). **Treat custom XMX kernels' output as suspect until you've cross-checked against a CPU reference.**

3. **SYCL backend is consistently 30-60% slower than Vulkan** for llama.cpp text generation on Battlemage. Per-kernel SYCL launch overhead and missing flash-attention / MMQ kernels are the named culprits. [llama.cpp discussion #12570](https://github.com/ggml-org/llama.cpp/discussions/12570). **For a from-scratch engine, going Level Zero direct (or even VK + cooperative-matrix) skips the SYCL runtime overhead entirely.**

4. **B580's `mul_mat_vec_q` only achieves 32% of theoretical bandwidth** (41.7 t/s vs ~128 t/s expected from 456 GB/s). The same kernel pattern carries to B70 - LLM decode is global-memory-bound and the existing public kernels do not stage A/B through 2D-block-loads optimally. [llama.cpp discussion #12570](https://github.com/ggml-org/llama.cpp/discussions/12570). **Use the 2D-block-load intrinsics; this is the single biggest perf knob for decode.**

5. **IPEX-LLM portable llama.cpp build is slower than upstream Vulkan llama.cpp** on the same B580 hardware - meaning Intel's own packaged stack is not the optimum. [ipex-llm issue #12991](https://github.com/intel/ipex-llm/issues/12991).

6. **Driver/kernel matrix is narrow** - the validated stack for compute-runtime 26.09 is *specifically* Ubuntu 24.04 + intel-graphics PPA kernel `6.14.0-1011-intel`. Stock distribution kernels (Ubuntu 24.04 LTS GA, Debian 12, RHEL 9.x default) are not validated and you will hit Xe-driver regressions and `force_probe` requirements. [compute-runtime 26.09 release page](https://github.com/intel/compute-runtime/releases/tag/26.09.37435.1).

7. **Mixed-dtype `dpas` is not exposed** at the OpenCL extension level (you can't do FP16×INT4 in one `dpas` call). For W4A16 you must dequantize INT4 to FP16 in registers - which costs SLM/regs. The reference is Intel's own [`sycl-tla`](https://github.com/intel/sycl-tla); replicate the pattern. (The `cl_intel_subgroup_matrix_multiply_accumulate` extension only enumerates same-dtype A/B combinations.)

8. **PCIe negotiates x8 on some AIB cards** even though the slot is x16 wired - the PMZFX benchmark rig saw "PCIe 5.0 x8 (negotiated)". For multi-GPU P2P traffic in vLLM-style tensor parallelism this halves your aggregation bandwidth. ([PMZFX hardware.md](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks/blob/master/hardware.md))

9. **MaxSun Dual-B60 cards are exposed as two separate GPUs**, not NVLink-style merged - this also applies to any multi-B70 build. There is **no NVLink/XGMI equivalent** on Battlemage. PCIe P2P is the only fast path, and the runtime supports it under the Battlematrix container ([StorageReview Battlematrix preview](https://www.storagereview.com/review/intel-arc-pro-b60-battlematrix-preview-192gb-of-vram-for-on-premise-ai)).

---

## 7. What I Could Not Verify

- **L2 cache size on BMG-G31 specifically.** ChipsAndCheese measured 18 MB on BMG-G21 (B580); Intel has not published the L2 size for BMG-G31 / B70 publicly. It is **plausibly higher** (because L2 is shared and the die has 60% more Xe-cores) but I have not found an authoritative number. Treat 18 MB as the **floor** when sizing tile/blocking strategies.
- **B70 BMG-G31 transistor count and die size.** Multiple secondary sources cite "~27.7B transistors" as a leak ([VideoCardz](https://videocardz.com/newz/intel-big-battlemage-bmg-g31-said-to-feature-27-7b-transistors-48-fewer-than-amd-navi-48)) but Intel has **not** officially published this. The B60's BMG-G21 is documented at 19.6B transistors / 272 mm² ([heise.de](https://www.heise.de/en/news/Intel-workstation-and-AI-graphics-cards-Arc-B50-and-B60-with-up-to-24-GByte-10388559.html)).
- **Matrix-path FP16 / BF16 / INT4 TFLOPS for B70 from Intel ARK directly.** ARK lists only `INT8 = 367 TOPS` and FP32. The FP16/BF16/INT4 numbers above are derived from VTune's documented per-clock ratios; they should be accurate but are not "spec-sheet" numbers.
- **Independent peer-reviewed benchmarks of XMX `dpas` real-world utilization on B70.** Phoronix has Linux benchmarks but not microbenchmarks of XMX achieved-vs-peak. Building your own roofline microbench should be priority-zero before kernel-tuning.
- **Actual oneAPI 2026.x release status.** As of 2026-04-25 the latest GA toolkit referenced in vendor pages is 2025.3.x; a "2026.0" toolkit may exist in early-access form but I could not find a public release page. Plan around 2025.3.
- **`bmg_g31`-specific SYCL Joint Matrix tile combinations beyond the OpenCL extension table.** Intel's 2025-version Joint Matrix doc page is access-gated; the [Khronos extension](https://github.com/khronosgroup/OpenCL-Docs/blob/main/extensions/cl_intel_subgroup_matrix_multiply_accumulate.asciidoc) is the most authoritative public reference for tile shapes today.
