# DeepSeek-V4 decode after the route fix: prefetch verdict and the CPU miss-split plan (2026-09-11, parked)

State: the non-abliterated ggml-org file runs at the abliterated file's level (pp 571 tok/s at 4096, tg 32.7 / 26.1
at 128 / 4096, serve prefill 463 at 14.6K) after 45f15f6 + 13f68e7. Founder (13:20): leave DeepSeek here.

## Prefetch verdict (results/deepseek4-prefetch-p1-2026-09-11/)
- ie-ds4-bench --prefetch on the two-card runtime issues 0 speculations: DeepSeek4TpRuntime::forward never sets
  spec_this_pass_ (only DeepSeek4Runtime::forward does), so prefetch has never been exercised under TP.
- The only predictor is 'this layer's ids for the next layer' (~1% accurate; 1.61x decode LOSS when it fired on
  2026-08-03, single runtime). docs/71 and the FreeToken paper agree: prediction cannot beat the link. Closed.
- Residency: 82 streaming slots/layer, 0 static (32% residency per card), hit 0.77 at 4K; the priority ranking
  has nothing to order. Decode = ~20 ms GPU + ~14.6 ms expert-miss DMA per card, alternating, per 38.5 ms token.

## Plan parked for later: CPU/PCIe miss split (GLM's q*, +44% there)
## Why this and not prefetch
Decode at 4K on the ggml-org file (two cards, expert-TP): per card ~20 ms GPU + ~14.6 ms expert-miss DMA per
38.5 ms token, alternating (a layer's misses are fetched between its router drain and its expert GEMMs); hit rate
0.77, 82 streaming slots/layer (32% residency), prefetch never useful (the only predictor is "same ids next layer",
~1% accurate — docs/71 and the paper it reviews both call predictive prefetch a dead end: "change how misses are
SERVED"). The link is the floor. GLM already serves a bandwidth-adaptive share of its misses on the CPU from the
pinned host banks (q* = m * B_PCIe / B_host; glm5next.cpp ~3360-3420, src/ops/cpu_moe_gemv.cpp) and gained +44%
(6.65 -> 9.60 tok/s, 2026-08-31) when its misses were the floor. DeepSeek has no CPU expert path: the CPU kernel is
Q4_K/Q5_K and DS4 experts are MXFP4 (planes qs [N][K/2] nibbles + e [N][K/32] e8m0 scales; every expert of this
card's half is pinned in the host arena: "256 of 256 experts/layer pinned").

## Arithmetic (pre-registered expectation, Pineapple until measured)
Per card per token: 59 misses x 6.69 MB = 395 MB over PCIe at ~27 GB/s = 14.6 ms. Host DRAM ~50 GB/s (measured
49 GB/s P-cores) shared by both cards' CPU shares. q* = B_PCIe / (B_PCIe + B_host) per link with both links live:
CPU takes ~48% of misses in bandwidth terms if the CPU GEMV runs at DRAM speed — which it will not (dequant + FMA on
8 P-cores; GLM's per-expert CPU cost was 0.56 ms for a ~14 MB Q4_K expert, i.e. ~25 GB/s effective). Honest
expectation: CPU absorbs 25-35% of misses -> DMA 14.6 -> ~10 ms per card -> token 38.5 -> ~34 ms = +12-15% decode
at 4K; more at deeper contexts where the hit rate is lower. Below +10% median over A/B/B/A = do not ship.

## Build (one phase, ~1-2 days with gates)
1. CPU MXFP4 expert GEMV (src/ops/cpu_moe_gemv.cpp): gate/up/down over the host-arena slot (this card's half:
   K = hidden, N = EFc), f32 activation in, exact e2m1 x 2^e8m0 LUT dequant, AVX2, OpenMP team; unit test vs
   ref::dequant + double dot on random experts (bit-exact vs a scalar reference of the same formula).
2. Split in moe_expert_grouped at T == 1 only: after the router drain, of the layer's m misses send round(q* m)
   to the CPU worker (persistent, core-pinned like GLM's CpuWorker; the same 8 P-cores are shared by both cards'
   runtimes — partition or serialize, measured), the rest through acquire() as today; the CPU computes
   w_e * down(swiglu(gate(x), up(x))) for its experts into a pinned host ring slot; ONE async H2D + accumulate into
   ws_moe_ before the reduction (the reduction sums the cards' halves as it does now). Static/hit experts unchanged.
3. Env: IE_DS4_QSTAR (default 0 = off until the gate passes, then the measured optimum), IE_DS4_CPU_CORES.
4. Numerics: CPU f32 vs GPU int-dot differ by design — NOT bit-identical. Gate = batch NLL unchanged (prefill is
   untouched: T > 1 never splits), stream PPL within +-0.5% of 3.9514, greedy 64-token text checked for coherence and
   determinism (identical across 3 runs), the 8-length stress clean, and the q* = 0 path byte-identical to today.
5. Measure: tg at 128 / 4096 / 16384 and the 16K serve, A/B/B/A at q* in {0.25, 0.4}; report CPU experts/token and
   ms each, DMA ms/token, hit rate.

## Not in this phase
Speculative decoding with the DSpark drafter (src/model/dspark_drafter.cpp exists but is wired to single-GPU
Qwen35-dense targets; a DS4 target-conditioned drafter + batched verify in the TP runtime is a larger build — the
structural way to turn per-token DMA into per-batch DMA); exact hash-router prefetch for blk.0-2 (~1 ms/token).

Pre-registered constraints from the gate's pre-read (must be in the criteria when this is picked up):
A. Merge folded into ds4_tp_reduce_host's host add (n = 8192 floats < the copy-engine threshold): no new H2D,
   no new host wait; the CPU's deadline is the layer's reduction. TP-only scope.
B. GO/NO-GO micro-bench BEFORE wiring: a half-expert (gate+up+swiglu+down over this card's 1024 intermediate
   columns, f32 in) single and two-concurrent on the pinned P-cores must fit ~0.35 ms => >= ~38 GB/s aggregate;
   plus the per-layer miss histogram (m in {0,1,2,3}) and an INTEGER split rule computed on it.
C. The split rule is a pure function of (layer, miss list, running byte counters) — never of timing; the CPU
   path is a third numerics variant (f32 vs Q8_1-quantized activations); gate = stream PPL within +-0.5%,
   q*=0 byte-identical, deterministic text; batch NLL unchanged by construction (regression check only).
D. IE_DS4_CPU_CORES pre-registered; measured with both cards live (two card threads + reducer + GEMV team).
Alternatives: DS4-conditioned DSpark drafter (per-batch DMA; drafter code is single-GPU Qwen35 today; file at
/media/<user>/Data1/models/keys-DeepSeekV4-Flash-DREAM-GGUF/DeepSeek-V4-Flash-0731-DSpark.gguf); exact hash-router
prefetch for blk.0-2 (~1 ms/token).
