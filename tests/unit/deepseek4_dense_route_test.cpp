// deepseek4_dense_route_test — the dense-weight PLACEMENT policy, host-only.
//
// WHAT DEFECT THIS TEST EXISTS FOR
// --------------------------------
// `Ds4DenseQuant` is a statement about what CONSUMES a weight, but until
// 2026-09-12 one of its values did not say what its call sites meant.
// `kKeepWide` reads as "hold this weight wide", and the three indexer
// projections used it for exactly that reason: their output feeds
// `ds4_indexer_score` -> `ds4_indexer_topk`, a DISCRETE top-k, where a perturbed
// score changes WHICH entries are attended. But `kKeepWide` only means
// "don't requantise" — it resolves to whatever the FILE stored. On an F16-dense
// GGUF that is fp16 and the intent holds; on the ggml-org Q8_0-dense GGUF it
// silently resolved to Q8_0, the precision the call site existed to avoid, and
// put those tensors on the per-element `dense_packed` kernel: 936.7 ms of a
// 4360.7 ms two-card prefill GPU budget, 7.43 ms/call against 0.29 for
// `dense_f16` (measured on the real model, docs/deepseek4/82).
//
// A policy whose meaning depends on the file it meets is a bug that no
// throughput number reports and no perplexity gate catches — both files load,
// both answer, one is 25x slower per call at a precision nobody asked for. So
// the rule is a pure function now, and this file pins it against the dtype/quant
// matrix rather than against one model that happens to be on disk.
//
// NO GPU, NO MODEL, NO ENV.
#include "ie/deepseek4.hpp"
#include "ie/dtype.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

int g_fail = 0;

const char* name(ie::Ds4DenseRoute r) {
    switch (r) {
        case ie::Ds4DenseRoute::kPacked:      return "packed";
        case ie::Ds4DenseRoute::kRequantised: return "requantised";
        case ie::Ds4DenseRoute::kFp16:        return "fp16";
    }
    return "?";
}
const char* name(ie::Ds4DenseQuant q) {
    switch (q) {
        case ie::Ds4DenseQuant::kAuto:     return "kAuto";
        case ie::Ds4DenseQuant::kKeepWide: return "kKeepWide";
        case ie::Ds4DenseQuant::kWideF16:  return "kWideF16";
    }
    return "?";
}

void want(const char* what, ie::DType dt, uint64_t K, uint64_t Ks, ie::Ds4DenseQuant q,
          ie::Ds4DenseRoute expect) {
    const ie::Ds4DenseRoute got = ie::ds4_dense_route(dt, K, Ks, q);
    const bool ok = got == expect;
    std::printf("%s %-58s %-10s K=%-5llu Ks=%-5llu -> %-11s (want %s)\n",
                ok ? "[ ok ]" : "[FAIL]", what, name(q),
                (unsigned long long)K, (unsigned long long)Ks, name(got), name(expect));
    if (!ok) ++g_fail;
}

}  // namespace

int main() {
    using ie::DType;
    using Q = ie::Ds4DenseQuant;
    using R = ie::Ds4DenseRoute;
    std::printf("=== deepseek4_dense_route_test — dense placement policy ===\n\n");

    // The requantiser is opt-out via IE_DS4_DENSE_Q8; this test states the
    // DEFAULT policy, so refuse to run under an override rather than print a
    // green line that describes a different engine.
    if (const char* e = std::getenv("IE_DS4_DENSE_Q8"); e && *e && std::string(e) != "1") {
        std::printf("[FAIL] IE_DS4_DENSE_Q8=%s overrides the policy under test\n", e);
        return 1;
    }

    // §1 THE DEFECT THIS TEST EXISTS FOR.  Same call site, same policy value,
    // two files: the answer must not change with the file when the call site
    // asked to be wide.  This is the assertion that would have caught it.
    std::puts("§1 a DISCRETE-decision site is fp16 on EVERY file (the 2026-09-12 defect)");
    want("indexer proj, F16-dense GGUF",  DType::kF16,  2048, 2048, Q::kWideF16, R::kFp16);
    want("indexer proj, Q8_0-dense GGUF", DType::kQ8_0, 2048, 2048, Q::kWideF16, R::kFp16);
    want("indexer proj, BF16-dense GGUF", DType::kBF16, 2048, 2048, Q::kWideF16, R::kFp16);
    want("indexer proj, Q6_K-dense GGUF", DType::kQ6_K, 2048, 2048, Q::kWideF16, R::kFp16);
    // ...and the control: kKeepWide is exactly the value that DOES follow the
    // file, which is why it is wrong for that site and right for `token_embd`.
    want("token_embd, F16-dense GGUF",    DType::kF16,  2048, 2048, Q::kKeepWide, R::kFp16);
    want("token_embd, Q8_0-dense GGUF",   DType::kQ8_0, 2048, 2048, Q::kKeepWide, R::kPacked);

    // §2 kAuto — the ordinary projections, whose output flows into arithmetic.
    std::puts("\n§2 kAuto requantises every dtype the requantiser accepts");
    for (auto dt : {DType::kF16, DType::kBF16, DType::kF32, DType::kQ8_0})
        want("ordinary projection", dt, 2048, 2048, Q::kAuto, R::kRequantised);
    // Q6_K is NOT requantised (it would be lossy) and HAS a packed form.
    want("ordinary projection, Q6_K", DType::kQ6_K, 2048, 2048, Q::kAuto, R::kPacked);

    // §3 THE K / Ks ASYMMETRY.  Whether a dtype has a packed form is a property
    // of the whole row; whether THIS CARD's column slice is a whole number of
    // blocks is a property of the slice.  A kCols split that lands mid-block
    // must not reach the requantiser, and must not silently keep a packed form
    // the decoder would mis-address.
    std::puts("\n§3 a column slice that is not a whole number of blocks");
    want("Q8_0, 4-way column split of K=128 (Ks=32, whole)",  DType::kQ8_0, 128, 32, Q::kAuto, R::kRequantised);
    want("Q8_0, column slice Ks=48 (not a whole 32-block)",   DType::kQ8_0, 128, 48, Q::kAuto, R::kPacked);
    want("F16, column slice Ks=48 (no packed form either)",   DType::kF16,  128, 48, Q::kAuto, R::kFp16);
    want("Q8_0, ragged slice at a kWideF16 site",             DType::kQ8_0, 128, 48, Q::kWideF16, R::kFp16);

    // §4 Degenerate lengths: K = 0 must not be requantised or packed.
    std::puts("\n§4 K = 0");
    for (auto q : {Q::kAuto, Q::kKeepWide, Q::kWideF16})
        want("zero-length contraction", DType::kQ8_0, 0, 0, q, R::kFp16);

    // §5 A dtype with NEITHER a packed form nor a requantised one falls to fp16
    // under every policy — the one branch that must not depend on the call site.
    // Q4_K is that dtype: `ds4_dense_keeps_packed` covers only Q8_0 and Q6_K, and
    // `ds4_dense_requantises` only the wide dtypes plus Q8_0.
    std::puts("\n§5 a dtype with neither form (Q4_K)");
    for (auto q : {Q::kAuto, Q::kKeepWide, Q::kWideF16})
        want("Q4_K source", DType::kQ4_K, 2048, 2048, q, R::kFp16);

    std::printf("\n%s\n", g_fail ? "FAILED" : "deepseek4_dense_route_test: OK");
    return g_fail ? 1 : 0;
}
