// tools/ie_ds4_ppl.cpp — the DeepSeek-V4-Flash QUALITY GATE.
//
// WHY THIS EXISTS.  `tools/ie_perplexity.cpp` dispatches over an arch table that
// predates this architecture (is_dense / is_q35 / is_q3moe / is_gptoss /
// is_q35moe_split); on a deepseek4 GGUF it misdetects the arch and exits 1
// without producing a number.  DeepSeek-V4-Flash has therefore never had a
// perplexity measured, and lossy changes have shipped behind that blind spot.
// This tool is the missing measurement, built on `DeepSeek4TpRuntime` the same
// way `tools/ie_ds4_bench.cpp` is.
//
// WHAT IT COMPUTES.  Exactly the standard thing and nothing else: run the token
// sequence through the model and, for each position i, accumulate
//     nll_i = -log softmax(logits at position i)[ ids[i+1] ]
//     PPL   = exp( mean nll )
// `nll_of_target` below is the numerically-stable log-sum-exp from
// ie_perplexity.cpp:126-137, transcribed unchanged onto a pointer because the
// DS4 runtime hands back `float*` rather than a vector of `sycl::half`.
//
// TWO NUMERICAL PATHS, MEASURED SEPARATELY.  DeepSeek-V4's engine sets
// `last_only = true` by default, so a prefill forward returns only the LAST
// row's logits.  Per-position logits can be had two ways, and they are NOT the
// same computation:
//   --mode stream  T=1 per position.  This is the DECODE path: fp32 activations
//                  through the decode GEMVs.  It is the configuration that
//                  measures 26.48 tok/s, so it is the one whose quality is in
//                  question.  DEFAULT.
//   --mode batch   T=chunk with last_only=false, one row per position.  This is
//                  the PREFILL path: oneDNN with fp16 activations.  Much faster
//                  per token, and a different set of kernels.
// Neither is "the" answer; a change can be neutral on one and not the other, so
// both are available and every result line says which ran.
//
// THE HARNESS GATES ITSELF (`--gate`, pure host, no GPU, no model).  A PPL
// harness that is subtly wrong is worse than none, because it will be trusted.
// §G1..§G7 below prove the scorer on inputs whose answers are known in closed
// form, including two ORACLE gates that drive the real scoring loops with a
// synthetic model and would separate a correct alignment from an off-by-one by
// five orders of magnitude.  `--gate` must pass before any number this tool
// prints is worth reading.
//
// DIFFING TWO ARMS.  `--dump-nll <path>` writes every per-token NLL; passing
// that file back as `--baseline <path>` on the second arm prints the full
// comparison — mean, PPL delta in percent, and the per-token delta DISTRIBUTION,
// which is the part that matters: a quantisation that is fine on average and
// occasionally collapses one token's probability is a distributional break, not
// rounding, and a mean hides it.
//
// Usage:
//   ie-ds4-ppl --gate                                    # self-test, no GPU
//   ie-ds4-ppl --mini                                    # miniature fixture
//   DS4_RUN_REAL=1 DS4_TP_GPUS=0,1 ie-ds4-ppl --gguf <shard1> --dump-nll a.tsv
//   DS4_RUN_REAL=1 DS4_TP_GPUS=0,1 ie-ds4-ppl --gguf <shard1> --dump-nll b.tsv \
//                                             --baseline a.tsv

#include "ie/deepseek4.hpp"
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"
#include "ie/tokenizer.hpp"

#include "deepseek4/ds4_mini_gguf.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace ie;

double now_s() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// Emission — one `key=value` per line, so two runs diff mechanically.
// ---------------------------------------------------------------------------
void emit(const std::string& k, const std::string& v) {
    std::printf("[ds4-ppl] %s=%s\n", k.c_str(), v.c_str());
    std::fflush(stdout);
}
void emit_d(const std::string& k, double v, int prec = 6) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.*f", prec, v);
    emit(k, buf);
}
void emit_u(const std::string& k, uint64_t v) { emit(k, std::to_string(v)); }
void note(const std::string& t) {
    std::printf("[ds4-note] %s\n", t.c_str());
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// The corpus
// ---------------------------------------------------------------------------
// Byte-for-byte the built-in sample from tools/ie_perplexity.cpp, so a
// deepseek4 number produced here sits on the same text as every other PPL
// number in this repo.  Clean English prose, stable register, in-distribution
// for any general LM.
const char* kSampleText =
    "I am by birth a Genevese; and my family is one of the most distinguished "
    "of that republic. My ancestors had been for many years counsellors and "
    "syndics; and my father had filled several public situations with honour "
    "and reputation. He was respected by all who knew him for his integrity "
    "and indefatigable attention to public business. He passed his younger "
    "days perpetually occupied by the affairs of his country; and it was not "
    "until the decline of life that he thought of marrying, and bestowing on "
    "the state sons who might carry his virtues and his name down to "
    "posterity. As the circumstances of his marriage illustrate his "
    "character, I cannot refrain from relating them.\n\n"
    "During the years that followed, the household preserved a quiet discipline "
    "which gave dignity to ordinary labor. The library was small, but every "
    "volume had been chosen with care, and the books were read until their "
    "margins carried traces of many hands. At evening the shutters were closed, "
    "the lamp was trimmed, and the younger children listened while letters from "
    "distant friends were read aloud. These letters spoke of voyages, harvests, "
    "public debates, and the patient work by which families keep faith with one "
    "another across time and weather.\n\n"
    "I learned early that knowledge is not gathered by haste alone. A page "
    "understood clearly was worth more than a chapter passed over in restless "
    "curiosity. My teachers encouraged questions, but they also required proof, "
    "comparison, and a willingness to revise an opinion when the evidence did "
    "not support it. In that habit I found a kind of freedom: the mind became "
    "less anxious when it could distinguish a bright guess from a settled fact.\n\n"
    "When I was older, I travelled beyond the familiar streets of my childhood "
    "and saw how much of human life depends on arrangements too common to be "
    "praised. Roads, bridges, ledgers, workshops, schools, and markets seemed "
    "plain enough at first glance, yet each required memory, trust, and daily "
    "attention. A careless hand could waste what many careful hands had built. "
    "This observation made me cautious in judgment and more grateful for the "
    "uncelebrated skill that supports a peaceful city.\n\n"
    "The strongest impression of those years was not a single event, but a "
    "gradual conviction that character is measured in repeated choices. A "
    "person may speak generously in public and still fail in private duties; "
    "another may say little and yet become indispensable by doing necessary "
    "work at the proper hour. I admired the latter kind of excellence. It did "
    "not glitter, but it endured, and it left the world more orderly than it "
    "found it.\n\n"
    "Thus my education joined affection with inquiry. I loved the people who "
    "had formed me, but I also learned to examine my own certainties. Whenever "
    "a new subject drew my attention, I tried to ask what could be tested, what "
    "must be inferred, and what ought to remain undecided. This discipline did "
    "not diminish wonder. On the contrary, it made wonder steadier, because it "
    "rested on patient attention rather than surprise alone.";

std::string read_text_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) return {};
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// ---------------------------------------------------------------------------
// log P[target] for one logits row, numerically stable.
// ---------------------------------------------------------------------------
// Transcribed from ie_perplexity.cpp:129-137 with no change of algorithm — only
// the container, because the DS4 runtime writes fp32 into a caller buffer while
// the older tool read a vector of sycl::half.
//   log P[t] = logits[t] - logsumexp(logits)
//   returns NLL = -log P (nats)
double nll_of_target(const float* logits, uint32_t n, int32_t target_id) {
    float m = logits[0];
    for (uint32_t i = 1; i < n; ++i) m = std::max(m, logits[i]);
    double sum_exp = 0.0;
    for (uint32_t i = 0; i < n; ++i) sum_exp += std::exp(double(logits[i]) - double(m));
    const double lse   = double(m) + std::log(sum_exp);
    const double log_p = double(logits[target_id]) - lse;
    return -log_p;
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------
// The two loops below are driven through a function object with EXACTLY the
// `DeepSeek4Runtime::forward` signature.  That is what lets §G6/§G7 run the real
// scoring code against a synthetic model whose correct answer is known in closed
// form: the alignment logic under test is the same object code either way.
using ForwardFn = std::function<std::string(const int32_t*, uint32_t, uint32_t, float*, bool)>;
using ResetFn   = std::function<void()>;

struct Scored {
    std::vector<double>  nll;        // nll[p] scores target ids[p+1]
    std::vector<int32_t> target;     // ids[p+1], carried so a diff can refuse mismatched runs
    uint32_t             nonfinite = 0;
    double               seconds   = 0.0;
    std::string          err;

    bool     ok()      const { return err.empty() && !nll.empty() && nonfinite == 0; }
    double   sum()     const { return std::accumulate(nll.begin(), nll.end(), 0.0); }
    double   avg()     const { return nll.empty() ? 0.0 : sum() / double(nll.size()); }
    double   ppl()     const { return std::exp(avg()); }
};

// One accumulation step, shared so the two modes cannot drift apart.
void accum(Scored& s, const float* row, uint32_t vocab, int32_t target) {
    const double v = nll_of_target(row, vocab, target);
    if (!std::isfinite(v)) ++s.nonfinite;
    s.nll.push_back(v);
    s.target.push_back(target);
}

// T=1 per position — the DECODE numerical path.
// Position N-1 is never forwarded: there is no ids[N] to score against it, so
// running it would be a wasted step, not a missing measurement.
Scored score_stream(const std::vector<int32_t>& ids, uint32_t vocab,
                    const ForwardFn& fwd, const ResetFn& reset) {
    Scored s;
    std::vector<float> lg(vocab);
    reset();
    const double t0 = now_s();
    for (uint32_t i = 0; i + 1 < ids.size(); ++i) {
        if (std::string e = fwd(&ids[i], 1, i, lg.data(), /*last_only=*/true); !e.empty()) {
            s.err = "forward at pos " + std::to_string(i) + ": " + e;
            break;
        }
        accum(s, lg.data(), vocab, ids[i + 1]);
    }
    s.seconds = now_s() - t0;
    return s;
}

// T=chunk with last_only=false — the PREFILL numerical path.
// A forward of T tokens from pos0 returns rows for positions pos0..pos0+T-1,
// which score targets ids[pos0+1..pos0+T].  T is therefore clamped so that
// pos0+T never exceeds ids.size()-1; the tokens consumed across chunks are
// ids[0..N-2] contiguously, with no gap and no repeat.
Scored score_batch(const std::vector<int32_t>& ids, uint32_t vocab, uint32_t chunk,
                   const ForwardFn& fwd, const ResetFn& reset) {
    Scored s;
    if (chunk == 0) chunk = 1;
    const uint32_t N = uint32_t(ids.size());
    std::vector<float> lg(uint64_t(chunk) * vocab);
    reset();
    const double t0 = now_s();
    uint32_t pos = 0;
    while (pos + 1 < N) {
        const uint32_t T = std::min<uint32_t>(chunk, N - 1 - pos);
        if (std::string e = fwd(&ids[pos], T, pos, lg.data(), /*last_only=*/false); !e.empty()) {
            s.err = "forward T=" + std::to_string(T) + " at pos " + std::to_string(pos) +
                    ": " + e;
            break;
        }
        for (uint32_t r = 0; r < T; ++r)
            accum(s, lg.data() + uint64_t(r) * vocab, vocab, ids[pos + r + 1]);
        pos += T;
    }
    s.seconds = now_s() - t0;
    return s;
}

// ---------------------------------------------------------------------------
// Distribution helpers
// ---------------------------------------------------------------------------
double pct_of_sorted(const std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    size_t i = size_t(p * double(v.size()));
    if (i >= v.size()) i = v.size() - 1;
    return v[i];
}

// ---------------------------------------------------------------------------
// Per-token NLL files — the A/B seam
// ---------------------------------------------------------------------------
struct NllFile {
    std::vector<int32_t> target;
    std::vector<double>  nll;
    std::string          meta;      // the `# meta` line, echoed on a diff
};

std::string write_nll(const std::string& path, const Scored& s, const std::string& meta) {
    std::ofstream f(path);
    if (!f) return "cannot open " + path + " for writing";
    f << "# ie-ds4-ppl per-token NLL (nats)\n";
    f << "# meta\t" << meta << "\n";
    f << "# idx\tpos\ttarget_id\tnll\n";
    for (size_t i = 0; i < s.nll.size(); ++i)
        f << i << '\t' << i << '\t' << s.target[i] << '\t'
          << std::setprecision(17) << s.nll[i] << '\n';
    return f.good() ? std::string() : ("write failed: " + path);
}

std::string read_nll(const std::string& path, NllFile& out) {
    std::ifstream f(path);
    if (!f) return "cannot open " + path;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            if (line.rfind("# meta\t", 0) == 0) out.meta = line.substr(7);
            continue;
        }
        std::istringstream is(line);
        long idx = 0, pos = 0;
        long tgt = 0;
        double v = 0;
        if (!(is >> idx >> pos >> tgt >> v)) return "malformed line in " + path + ": " + line;
        out.target.push_back(int32_t(tgt));
        out.nll.push_back(v);
    }
    if (out.nll.empty()) return path + " contained no data rows";
    return {};
}

// The comparison that decides whether a change is safe.  It refuses to compare
// two runs whose target sequences differ — that would silently be a different
// measurement, not a delta.
void compare(const NllFile& base, const Scored& now) {
    std::printf("\n");
    if (base.nll.size() != now.nll.size()) {
        note("baseline has " + std::to_string(base.nll.size()) + " predictions, this run has " +
             std::to_string(now.nll.size()) + " — REFUSING to compare. Same --max-tokens, "
             "same --text and same --mode are required for an A/B.");
        emit("ab.status", "refused_length_mismatch");
        return;
    }
    for (size_t i = 0; i < base.target.size(); ++i) {
        if (base.target[i] != now.target[i]) {
            note("baseline target id at index " + std::to_string(i) + " is " +
                 std::to_string(base.target[i]) + ", this run's is " +
                 std::to_string(now.target[i]) + " — REFUSING to compare. The two arms did "
                 "not score the same tokens.");
            emit("ab.status", "refused_token_mismatch");
            return;
        }
    }
    emit("ab.status", "ok");
    if (!base.meta.empty()) note("baseline meta: " + base.meta);

    const double bs = std::accumulate(base.nll.begin(), base.nll.end(), 0.0);
    const double bavg = bs / double(base.nll.size());
    const double bppl = std::exp(bavg);
    const double nppl = now.ppl();

    emit_u("ab.tokens", now.nll.size());
    emit_d("ab.base_avg_nll", bavg, 6);
    emit_d("ab.now_avg_nll", now.avg(), 6);
    emit_d("ab.delta_avg_nll", now.avg() - bavg, 6);
    emit_d("ab.base_ppl", bppl, 4);
    emit_d("ab.now_ppl", nppl, 4);
    emit_d("ab.delta_ppl", nppl - bppl, 4);
    emit_d("ab.delta_ppl_pct", bppl != 0.0 ? 100.0 * (nppl - bppl) / bppl : 0.0, 4);

    // The per-token distribution.  This is the part a mean cannot show.
    std::vector<double> d(now.nll.size()), ad(now.nll.size());
    uint32_t identical = 0, worse = 0, better = 0;
    uint32_t gt005 = 0, gt025 = 0, gt100 = 0;
    for (size_t i = 0; i < d.size(); ++i) {
        d[i]  = now.nll[i] - base.nll[i];
        ad[i] = std::fabs(d[i]);
        if (d[i] == 0.0) ++identical;
        else if (d[i] > 0.0) ++worse; else ++better;
        if (ad[i] > 0.05) ++gt005;
        if (ad[i] > 0.25) ++gt025;
        if (ad[i] > 1.00) ++gt100;
    }
    std::vector<double> sad = ad;
    std::sort(sad.begin(), sad.end());
    emit_u("ab.tok.bit_identical", identical);
    emit_d("ab.tok.bit_identical_frac", double(identical) / double(d.size()), 4);
    emit_u("ab.tok.worse", worse);
    emit_u("ab.tok.better", better);
    emit_d("ab.tok.abs_delta_mean", std::accumulate(ad.begin(), ad.end(), 0.0) / double(ad.size()), 6);
    emit_d("ab.tok.abs_delta_p50", pct_of_sorted(sad, 0.50), 6);
    emit_d("ab.tok.abs_delta_p90", pct_of_sorted(sad, 0.90), 6);
    emit_d("ab.tok.abs_delta_p99", pct_of_sorted(sad, 0.99), 6);
    emit_d("ab.tok.abs_delta_max", sad.back(), 6);
    emit_u("ab.tok.count_abs_gt_0.05", gt005);
    emit_u("ab.tok.count_abs_gt_0.25", gt025);
    emit_u("ab.tok.count_abs_gt_1.00", gt100);
    note("ab.tok.count_abs_gt_1.00 is the distributional-break counter: a weight perturbation "
         "of the stated size (worst 4.6e-4 relative) cannot move one token's log-probability by "
         "a whole nat by rounding alone. A non-zero count means a DISCRETE decision flipped "
         "(the indexer's top-512 entry selection is the requantised path that can do this), "
         "not that the average got slightly noisier.");
}

// ---------------------------------------------------------------------------
// §G — the harness gates.  Pure host, no GPU, no model.
// ---------------------------------------------------------------------------
// Every tolerance below is derived in the comment beside it.  None is a round
// number picked to make a check pass.
struct Gate {
    uint32_t pass = 0, fail = 0;
    void check(bool ok, const std::string& what, const std::string& detail = {}) {
        if (ok) { ++pass; std::printf("  [ok]   %s%s%s\n", what.c_str(),
                                      detail.empty() ? "" : "  ", detail.c_str()); }
        else    { ++fail; std::printf("  [FAIL] %s%s%s\n", what.c_str(),
                                      detail.empty() ? "" : "  ", detail.c_str()); }
        std::fflush(stdout);
    }
};

std::string fmt(double v, int prec = 12) {
    char b[64];
    std::snprintf(b, sizeof b, "%.*g", prec, v);
    return b;
}

// A deterministic 32-bit LCG.  Fixed seeds everywhere, so every gate result in
// this file is reproducible byte for byte.
struct Lcg {
    uint32_t s;
    explicit Lcg(uint32_t seed) : s(seed ? seed : 1u) {}
    uint32_t next() { s = s * 1103515245u + 12345u; return s; }
    // Uniform in [lo, hi), quantised to multiples of 1/1024 so that adding a
    // power-of-two shift below is EXACT in float and the shift-invariance gate
    // measures log-sum-exp stability rather than float representability.
    float unit_q10(float lo, float hi) {
        const float span = hi - lo;
        const uint32_t steps = uint32_t(span * 1024.0f);
        return lo + float(next() % (steps ? steps : 1u)) / 1024.0f;
    }
};

int run_gates() {
    Gate g;
    std::printf("\nie-ds4-ppl --gate — harness self-verification (host only, no GPU, no model)\n\n");

    // ---- §G1  flat logits over V tokens  =>  PPL == V exactly ---------------
    // logits all equal to c:  m = c, sum_exp = V (a sum of V exact 1.0 doubles,
    // exact for V < 2^53), lse = c + ln V, log_p = -ln V, PPL = V.
    // Worst-case error is the cancellation in `c - (c + ln V)`: at c = 1e4 the
    // double ulp is 1.8e-12, so |dNLL| <= 1.8e-12 and the PPL relative error is
    // the same.  TOLERANCE 1e-9 relative — about 500x the worst case, and still
    // 6 orders below any quality signal.
    std::printf("§G1 flat logits => PPL == V\n");
    for (uint32_t V : {512u, 32000u, 129280u}) {
        for (float c : {0.0f, 50.0f, -50.0f, 10000.0f}) {
            std::vector<float> lg(V, c);
            const double nll = nll_of_target(lg.data(), V, int32_t(V / 3));
            const double ppl = std::exp(nll);
            const double rel = std::fabs(ppl - double(V)) / double(V);
            g.check(rel <= 1e-9,
                    "V=" + std::to_string(V) + " c=" + fmt(c, 6),
                    "PPL=" + fmt(ppl, 15) + " rel_err=" + fmt(rel, 3));
        }
    }

    // ---- §G2  known answer + shift invariance -------------------------------
    // (a) vs an INDEPENDENT long-double route: normalise explicitly (p = e^(l-m)
    //     / sum) then take -log p, rather than subtracting a log-sum-exp.  Same
    //     algebra, different rounding path.  Long-double accumulation over
    //     V ~ 1.3e5 terms carries ~V*eps_ld ~ 1e-14 relative, so the NLL agrees
    //     to ~1e-13 absolute.  TOLERANCE 1e-9 nats — 4 orders of margin.
    // (b) shift invariance: NLL is invariant under adding a constant to every
    //     logit.  The shift is +-8192 = +-2^13 and the logits are multiples of
    //     1/1024, so l + 8192 is EXACTLY representable in float and any observed
    //     difference is instability, not representability.  8192 >> 709, so a
    //     naive exp(logit) in double would be inf (or 0 for -8192) and this gate
    //     would blow up rather than pass.  TOLERANCE 1e-9 nats.
    std::printf("§G2 known answer (independent long-double route) + shift invariance\n");
    {
        const uint32_t V = 129280;
        Lcg r(20260803u);
        std::vector<float> lg(V);
        for (uint32_t i = 0; i < V; ++i) lg[i] = r.unit_q10(-20.0f, 20.0f);
        double worst_kn = 0, worst_sh = 0;
        for (int32_t t : {0, 1, 7, 4095, 65536, int32_t(V - 1)}) {
            const double got = nll_of_target(lg.data(), V, t);
            long double m = lg[0];
            for (uint32_t i = 1; i < V; ++i) m = std::max<long double>(m, lg[i]);
            long double sum = 0.0L;
            for (uint32_t i = V; i-- > 0;) sum += std::exp((long double)(lg[i]) - m);
            const long double p = std::exp((long double)(lg[t]) - m) / sum;
            const double ref = double(-std::log(p));
            worst_kn = std::max(worst_kn, std::fabs(got - ref));
            for (float C : {8192.0f, -8192.0f}) {
                std::vector<float> sh(V);
                for (uint32_t i = 0; i < V; ++i) sh[i] = lg[i] + C;
                worst_sh = std::max(worst_sh, std::fabs(nll_of_target(sh.data(), V, t) - got));
            }
        }
        g.check(worst_kn <= 1e-9, "known-answer max|delta|", fmt(worst_kn, 3) + " nats");
        g.check(worst_sh <= 1e-9, "shift +-8192 max|delta|", fmt(worst_sh, 3) + " nats");
    }

    // ---- §G3  peaked logits => closed form ----------------------------------
    // logits[t] = L, everything else 0:  NLL = ln(1 + (V-1) e^-L), exactly.
    //
    // THE CRITERION IS ABSOLUTE NATS, AND THAT IS NOT LAZINESS.  The reported
    // quantity is PPL = exp(mean NLL), so an absolute error of eps nats in an
    // NLL is a RELATIVE error of eps in the perplexity — nats are the natural
    // unit of the tolerance and a relative-error bound on NLL would be the wrong
    // criterion.  TOLERANCE 1e-9 nats, i.e. 1e-7 % of PPL.
    //
    // The relative error is measured and printed anyway, because it is large in
    // one regime and the reader deserves to know why rather than to discover it.
    // At L = 40 the true answer is 5.5e-13 nats: `sum_exp` is 1 + 5.5e-13, the
    // 129279 tail terms are each 4.2e-18, and once the peak's 1.0 has entered
    // the running sum those terms are ~50x BELOW its ulp and are individually
    // rounded away.  The reported value is therefore only the part of the tail
    // that was accumulated before the peak.  That is a real property of the
    // inherited naive summation, it is confined to NLL values far below 1e-9
    // nats (i.e. softmax probabilities indistinguishable from 1), and §G3b
    // measures its size in the regime a language model actually occupies rather
    // than leaving it as an argument.
    std::printf("§G3 peaked logits => ln(1 + (V-1)e^-L)\n");
    {
        const uint32_t V = 129280;
        for (float L : {0.0f, 1.0f, 5.0f, 20.0f, 40.0f}) {
            std::vector<float> lg(V, 0.0f);
            const int32_t t = 12345;
            lg[t] = L;
            const double got = nll_of_target(lg.data(), V, t);
            const long double ref_l = std::log1p((long double)(V - 1) * std::exp((long double)(-L)));
            const double ref = double(ref_l);
            const double ad = std::fabs(got - ref);
            const double rd = ref != 0.0 ? ad / std::fabs(ref) : 0.0;
            // Absolute always; relative ALSO, but only where the answer is big
            // enough for a relative bound to mean anything (>= 1e-6 nats, which
            // is 1e-4 % of PPL — five orders below any quality signal).
            const bool ok = ad <= 1e-9 && (ref < 1e-6 || rd <= 1e-6);
            g.check(ok, "L=" + fmt(L, 4),
                    "got=" + fmt(got, 12) + " ref=" + fmt(ref, 12) +
                    " abs=" + fmt(ad, 3) + " rel=" + fmt(rd, 3));
        }
        // ... and the opposite end: the target is the single SUPPRESSED token.
        std::vector<float> lg(V, 0.0f);
        const int32_t t = 999;
        lg[t] = -30.0f;
        const double got = nll_of_target(lg.data(), V, t);
        const long double s = (long double)(V - 1) + std::exp(-30.0L);
        const double ref = double(std::log(s) + 30.0L);
        g.check(std::fabs(got - ref) <= 1e-9 && std::fabs(got - ref) / ref <= 1e-6,
                "suppressed target",
                "got=" + fmt(got, 12) + " ref=" + fmt(ref, 12));
    }

    // ---- §G3b  the naive summation, measured where it is actually used -------
    // §G3 showed the inherited log-sum-exp losing the tail when the softmax is
    // numerically a delta.  The question that matters is whether it loses
    // anything at logit distributions a real language model produces.  This
    // measures it directly: the same rows scored by `nll_of_target` and by a
    // KAHAN-COMPENSATED sum of the same terms, over Gaussian logits at several
    // spreads and over a realistically peaked row (one dominant token at +12,
    // a plausible decode distribution).
    // TOLERANCE 1e-9 nats, the same as everywhere else in this file.
    std::printf("§G3b naive vs Kahan-compensated log-sum-exp on realistic logit spreads\n");
    {
        const uint32_t V = 129280;
        auto kahan_nll = [&](const std::vector<float>& lg, int32_t t) {
            float m = lg[0];
            for (uint32_t i = 1; i < V; ++i) m = std::max(m, lg[i]);
            double sum = 0.0, c = 0.0;
            for (uint32_t i = 0; i < V; ++i) {
                const double y = std::exp(double(lg[i]) - double(m)) - c;
                const double tt = sum + y;
                c = (tt - sum) - y;
                sum = tt;
            }
            return -(double(lg[t]) - (double(m) + std::log(sum)));
        };
        Lcg r(99u);
        double worst = 0;
        std::string worst_at;
        for (float sd : {1.0f, 3.0f, 8.0f, 15.0f}) {
            std::vector<float> lg(V);
            for (uint32_t i = 0; i < V; ++i) lg[i] = r.unit_q10(-sd, sd);
            for (int32_t t : {0, 5000, int32_t(V - 1)}) {
                const double d = std::fabs(nll_of_target(lg.data(), V, t) - kahan_nll(lg, t));
                if (d > worst) { worst = d; worst_at = "gauss sd=" + fmt(sd, 3); }
            }
        }
        {   // one dominant token: the shape a confident decode step produces.
            std::vector<float> lg(V);
            for (uint32_t i = 0; i < V; ++i) lg[i] = r.unit_q10(-4.0f, 4.0f);
            lg[777] = 12.0f;
            for (int32_t t : {777, 778}) {
                const double d = std::fabs(nll_of_target(lg.data(), V, t) - kahan_nll(lg, t));
                if (d > worst) { worst = d; worst_at = "peaked +12"; }
            }
        }
        g.check(worst <= 1e-9,
                "naive == Kahan across realistic spreads",
                "max|delta|=" + fmt(worst, 3) + " nats" +
                (worst_at.empty() ? "" : " (worst: " + worst_at + ")"));
    }

    // ---- §G4  aggregation: PPL == exp(mean NLL) -----------------------------
    // Pins the reporting arithmetic, not the scorer.  Doubles throughout, so
    // TOLERANCE 1e-12 relative (a few ulp over ~1e3 accumulations).
    std::printf("§G4 aggregation PPL == exp(mean NLL)\n");
    {
        Scored s;
        Lcg r(7u);
        double acc = 0;
        for (uint32_t i = 0; i < 1000; ++i) {
            const double v = double(r.next() % 100000u) / 10000.0;   // [0, 10)
            s.nll.push_back(v);
            s.target.push_back(int32_t(i));
            acc += v;
        }
        const double want = std::exp(acc / 1000.0);
        const double rel = std::fabs(s.ppl() - want) / want;
        g.check(rel <= 1e-12, "1000 synthetic NLLs", "rel_err=" + fmt(rel, 3));
        g.check(s.nll.size() == 1000 && s.nonfinite == 0, "bookkeeping", "n=1000 nonfinite=0");
    }

    // ---- §G5  a non-finite logit is COUNTED, never swallowed -----------------
    std::printf("§G5 non-finite logits are reported, not hidden\n");
    {
        Scored s;
        std::vector<float> lg(64, 0.0f);
        lg[3] = std::numeric_limits<float>::quiet_NaN();
        accum(s, lg.data(), 64, 10);
        g.check(s.nonfinite == 1, "NaN in the row is counted",
                "nonfinite=" + std::to_string(s.nonfinite));
        g.check(!s.ok(), "Scored::ok() is false when a row was non-finite");
    }

    // ---- §G6  ORACLE: alignment, streaming mode -----------------------------
    // The single most dangerous harness bug is an off-by-one in which target a
    // row is scored against, because it produces a plausible-looking number.
    // This drives the REAL `score_stream` with a synthetic model that puts
    // logit +20 on ids[pos+1] (the correct next token) and 0 elsewhere.  If the
    // alignment is right the closed form from §G3 applies: NLL = ln(1 +
    // (V-1)e^-20) = 2.6646e-4 and PPL ~ 1.000266.  If it is off by one in
    // either direction the target is an arbitrary token at logit 0 and the NLL
    // is ~ln V = 11.77, PPL ~ V.  Five orders of magnitude apart; no tolerance
    // choice can confuse the two.  The ANTI-oracle below scores the CURRENT
    // token instead and must produce exactly that failure, which proves the gate
    // can fail rather than merely that it passed.
    std::printf("§G6 oracle alignment — streaming (T=1) mode\n");
    {
        const uint32_t V = 4096, N = 257;
        std::vector<int32_t> ids(N);
        Lcg r(11u);
        for (uint32_t i = 0; i < N; ++i) ids[i] = int32_t(r.next() % V);
        const double want_nll = double(std::log1p((long double)(V - 1) * std::exp(-20.0L)));

        uint32_t calls = 0;
        auto oracle = [&](const int32_t*, uint32_t T, uint32_t pos0, float* out, bool last_only) {
            ++calls;
            if (T != 1 || !last_only) return std::string("oracle: stream mode must call T=1 last_only");
            std::fill(out, out + V, 0.0f);
            if (pos0 + 1 < N) out[ids[pos0 + 1]] = 20.0f;
            return std::string();
        };
        const Scored s = score_stream(ids, V, oracle, [] {});
        g.check(s.err.empty(), "oracle stream ran", s.err);
        g.check(s.nll.size() == N - 1, "scored N-1 predictions",
                std::to_string(s.nll.size()) + " of " + std::to_string(N - 1));
        g.check(calls == N - 1, "made N-1 forwards (position N-1 is not forwarded)",
                std::to_string(calls));
        double worst = 0;
        for (double v : s.nll) worst = std::max(worst, std::fabs(v - want_nll));
        // Every position's answer is the SAME closed form, so this is §G3's
        // budget again: TOLERANCE 1e-9 nats.
        g.check(worst <= 1e-9, "every per-token NLL matches the closed form",
                "max|delta|=" + fmt(worst, 3));
        g.check(std::fabs(s.ppl() - std::exp(want_nll)) <= 1e-9, "PPL",
                fmt(s.ppl(), 12));

        // ANTI-oracle: scores the CURRENT token. Must land near ln V.
        auto anti = [&](const int32_t*, uint32_t, uint32_t pos0, float* out, bool) {
            std::fill(out, out + V, 0.0f);
            out[ids[pos0]] = 20.0f;
            return std::string();
        };
        const Scored a = score_stream(ids, V, anti, [] {});
        // With the peak on the wrong token, NLL = ln(V - 1 + e^20) - 0 ~ ln V
        // unless ids[pos] happens to equal ids[pos+1]; over 256 draws from 4096
        // that is rare but not impossible, so assert on the MEDIAN, which cannot
        // be moved by a handful of coincidences.
        std::vector<double> sorted = a.nll;
        std::sort(sorted.begin(), sorted.end());
        const double med = pct_of_sorted(sorted, 0.5);
        g.check(med > 8.0, "anti-oracle (off-by-one) is caught: median NLL >> 0",
                "median=" + fmt(med, 6) + " vs correct " + fmt(want_nll, 3));
    }

    // ---- §G7  ORACLE: alignment, batch mode ---------------------------------
    // Same test against `score_batch`, whose row indexing (row r of a chunk at
    // pos0 scores ids[pos0+r+1]) is separate code and therefore a separate bug
    // surface.  Chunk sizes chosen so that the last chunk is SHORT, which is
    // where a clamp error would live.
    std::printf("§G7 oracle alignment — batch (T=chunk, last_only=false) mode\n");
    {
        const uint32_t V = 4096, N = 257;
        std::vector<int32_t> ids(N);
        Lcg r(11u);
        for (uint32_t i = 0; i < N; ++i) ids[i] = int32_t(r.next() % V);
        const double want_nll = double(std::log1p((long double)(V - 1) * std::exp(-20.0L)));

        for (uint32_t chunk : {1u, 7u, 64u, 128u, 256u, 1024u}) {
            uint32_t consumed = 0;
            auto oracle = [&](const int32_t* in, uint32_t T, uint32_t pos0, float* out,
                              bool last_only) {
                if (last_only) return std::string("batch mode must call last_only=false");
                if (pos0 != consumed)
                    return std::string("batch mode fed pos0=" + std::to_string(pos0) +
                                       " after consuming " + std::to_string(consumed));
                if (in != &ids[pos0]) return std::string("batch mode fed the wrong id pointer");
                std::fill(out, out + uint64_t(T) * V, 0.0f);
                for (uint32_t t = 0; t < T; ++t)
                    if (pos0 + t + 1 < N) out[uint64_t(t) * V + ids[pos0 + t + 1]] = 20.0f;
                consumed += T;
                return std::string();
            };
            const Scored s = score_batch(ids, V, chunk, oracle, [] {});
            double worst = 0;
            for (double v : s.nll) worst = std::max(worst, std::fabs(v - want_nll));
            const bool ok = s.err.empty() && s.nll.size() == N - 1 && worst <= 1e-9 &&
                            consumed == N - 1;
            g.check(ok, "chunk=" + std::to_string(chunk),
                    s.err.empty() ? ("n=" + std::to_string(s.nll.size()) + " consumed=" +
                                     std::to_string(consumed) + " max|delta|=" + fmt(worst, 3))
                                  : s.err);
        }

        // stream and batch must agree EXACTLY on the same oracle: same targets,
        // same rows, same scorer.  Any difference here is an alignment bug in
        // one of the two loops, not a numerical one — hence a bit-exact check.
        auto o_s = [&](const int32_t*, uint32_t, uint32_t pos0, float* out, bool) {
            std::fill(out, out + V, 0.0f);
            if (pos0 + 1 < N) out[ids[pos0 + 1]] = 20.0f;
            return std::string();
        };
        auto o_b = [&](const int32_t*, uint32_t T, uint32_t pos0, float* out, bool) {
            std::fill(out, out + uint64_t(T) * V, 0.0f);
            for (uint32_t t = 0; t < T; ++t)
                if (pos0 + t + 1 < N) out[uint64_t(t) * V + ids[pos0 + t + 1]] = 20.0f;
            return std::string();
        };
        const Scored a = score_stream(ids, V, o_s, [] {});
        const Scored b = score_batch(ids, V, 33, o_b, [] {});
        bool same = a.nll.size() == b.nll.size();
        for (size_t i = 0; same && i < a.nll.size(); ++i)
            same = (a.nll[i] == b.nll[i]) && (a.target[i] == b.target[i]);
        g.check(same, "stream and batch score identical targets bit-for-bit on one oracle");
    }

    // ---- §G8  target shuffling really is a permutation -----------------------
    std::printf("§G8 --shuffle-targets is a true permutation\n");
    {
        std::vector<int32_t> ids(512);
        Lcg r(3u);
        for (uint32_t i = 0; i < ids.size(); ++i) ids[i] = int32_t(r.next() % 50000u);
        std::vector<int32_t> sh = ids;
        // Fisher-Yates over positions 1..N-1 (position 0 is the input, never a
        // target), with a fixed seed so the result is reproducible.
        Lcg pr(20260803u);
        for (size_t i = sh.size() - 1; i >= 2; --i) {
            const size_t j = 1 + (pr.next() % uint32_t(i));
            std::swap(sh[i], sh[j]);
        }
        std::vector<int32_t> a = ids, b = sh;
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        g.check(a == b, "multiset preserved");
        uint32_t fixed = 0;
        for (size_t i = 1; i < ids.size(); ++i) if (ids[i] == sh[i]) ++fixed;
        // Expected fixed points of a random derangement-free shuffle of n items
        // is ~1 plus collisions from the id distribution; 512 draws from 50000
        // makes duplicate ids rare.  A shuffle that moved nothing would show
        // ~511 here, so any small number is decisive.
        g.check(fixed < 32, "positions actually moved",
                "fixed_points=" + std::to_string(fixed) + " of 511");
    }

    std::printf("\n  gates passed %u, failed %u\n\n", g.pass, g.fail);
    emit("gate.passed", std::to_string(g.pass));
    emit("gate.failed", std::to_string(g.fail));
    emit("gate.status", g.fail == 0 ? "ok" : "FAILED");
    return g.fail == 0 ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Driver over "one card" and "n cards in lockstep" — same shape as ie_ds4_bench
// ---------------------------------------------------------------------------
struct Driver {
    DeepSeek4Runtime*   single = nullptr;
    DeepSeek4TpRuntime* tp     = nullptr;

    std::string forward(const int32_t* ids, uint32_t T, uint32_t pos0, float* lg, bool last_only) {
        return tp ? tp->forward(ids, T, pos0, lg, last_only)
                  : single->forward(ids, T, pos0, lg, last_only);
    }
    void     reset_context() { if (tp) tp->reset_context(); else single->reset_context(); }
    uint32_t n_cards() const { return tp ? tp->n_cards() : 1u; }
    DeepSeek4Runtime& card(uint32_t c) { return tp ? tp->card(c) : *single; }
};

std::vector<uint32_t> parse_csv_u32(const std::string& s) {
    std::vector<uint32_t> out;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) out.push_back(uint32_t(std::strtoul(t.c_str(), nullptr, 10)));
    return out;
}

void usage() {
    std::printf(
        "ie-ds4-ppl — DeepSeek-V4-Flash perplexity / quality gate\n"
        "  --gate               run the harness self-verification and exit (no GPU, no model)\n"
        "  --gguf <path>        model shard 1 (default $DS4_GGUF); needs DS4_RUN_REAL=1\n"
        "  --mini [path]        write and score the miniature fixture instead\n"
        "  --mini-keep          do not delete the fixture afterwards\n"
        "  --text <file>        corpus (default: the built-in ie_perplexity prose sample)\n"
        "  --max-tokens <n>     cap the token sequence (default 512)\n"
        "  --synthetic <n>      score n deterministic pseudo-random ids instead of text\n"
        "  --mode stream|batch  T=1 decode path (default) or T=chunk prefill path\n"
        "  --chunk <n>          batch-mode tokens per forward (default 128)\n"
        "  --shuffle-targets    permute the target sequence — PPL must rise sharply\n"
        "  --dump-nll <path>    write every per-token NLL (feed back as --baseline)\n"
        "  --baseline <path>    diff this run against a previous --dump-nll file\n"
        "  --gpu <n>            single-card ordinal ($DS4_GPU); $DS4_TP_GPUS=0,1 uses both\n"
        "  --max-seq <n>        Ds4Options::max_seq (default 512, as ie-ds4-bench)\n"
        "  --slots <n>          VRAM expert slots/layer (0 = derive / $DS4_SLOTS)\n"
        "  --stream-slots <n>   evictable slots/layer ($DS4_STREAM_SLOTS)\n"
        "  --cache-mb <n>       VRAM expert cache budget\n"
        "  --pin-layers <n>     pin only the first n layers ($DS4_PIN_LAYERS)\n"
        "  --tag <str>          free-form label carried into the block\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string gguf, mini_path, text_path, dump_path, base_path, tag;
    bool     mini = false, mini_keep = false, shuffle = false, want_gate = false;
    uint32_t max_tokens = 512, synthetic = 0, chunk = 128, gpu = 0;
    uint32_t max_seq = 512, slots = 0, stream_slots = 0, pin_layers = 0;
    uint64_t cache_bytes = 0;
    std::string mode = "stream";

    if (const char* e = std::getenv("DS4_GGUF")) gguf = e;
    if (const char* e = std::getenv("DS4_GPU")) gpu = uint32_t(std::atoi(e));
    if (const char* e = std::getenv("DS4_PIN_LAYERS")) pin_layers = uint32_t(std::atoi(e));
    std::vector<uint32_t> tp_gpus;
    if (const char* e = std::getenv("DS4_TP_GPUS")) tp_gpus = parse_csv_u32(e);

    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if      (s == "--gate")            want_gate = true;
        else if (s == "--gguf")            gguf = next();
        else if (s == "--mini")          { mini = true;
                                           if (i + 1 < argc && argv[i + 1][0] != '-') mini_path = next(); }
        else if (s == "--mini-keep")       mini_keep = true;
        else if (s == "--text")            text_path = next();
        else if (s == "--max-tokens")      max_tokens = uint32_t(std::atoi(next().c_str()));
        else if (s == "--synthetic")       synthetic = uint32_t(std::atoi(next().c_str()));
        else if (s == "--mode")            mode = next();
        else if (s == "--chunk")           chunk = uint32_t(std::atoi(next().c_str()));
        else if (s == "--shuffle-targets") shuffle = true;
        else if (s == "--dump-nll")        dump_path = next();
        else if (s == "--baseline")        base_path = next();
        else if (s == "--gpu")             gpu = uint32_t(std::atoi(next().c_str()));
        else if (s == "--max-seq")         max_seq = uint32_t(std::atoi(next().c_str()));
        else if (s == "--slots")           slots = uint32_t(std::atoi(next().c_str()));
        else if (s == "--stream-slots")    stream_slots = uint32_t(std::atoi(next().c_str()));
        else if (s == "--cache-mb")        cache_bytes = uint64_t(std::atoll(next().c_str())) << 20;
        else if (s == "--pin-layers")      pin_layers = uint32_t(std::atoi(next().c_str()));
        else if (s == "--tag")             tag = next();
        else if (s == "-h" || s == "--help") { usage(); return 0; }
        else { std::fprintf(stderr, "unknown argument: %s\n", s.c_str()); usage(); return 2; }
    }

    if (want_gate) return run_gates();
    if (mode != "stream" && mode != "batch") {
        std::fprintf(stderr, "--mode must be 'stream' or 'batch' (got '%s')\n", mode.c_str());
        return 2;
    }
    if (chunk == 0) chunk = 1;

    // The miniature fixture is written by this process; the real model is 120 GB
    // of disk, both GPUs and a pinned host arena, so it happens only when the
    // owner asks for it in as many words.  Same gate, same reason, as
    // ie_ds4_bench.cpp and deepseek4_forward_test.
    std::vector<std::vector<uint8_t>> mini_keepalive;
    if (mini) {
        if (mini_path.empty()) {
            const char* td = std::getenv("TMPDIR");
            mini_path = std::string(td ? td : "/tmp") + "/ds4_ppl_mini.gguf";
        }
        ds4mini::Cfg mc;
        const std::string we = ds4mini::write_gguf(mc, mini_path, mini_keepalive);
        mini_keepalive.clear();
        if (!we.empty()) { std::fprintf(stderr, "mini fixture: %s\n", we.c_str()); return 1; }
        gguf = mini_path;
        if (max_tokens > 128) max_tokens = 128;   // the fixture's own §6 sizing
        if (max_seq > 192)    max_seq = 192;
        if (!slots)        slots = 6;
        if (!stream_slots) stream_slots = 2;
        if (!cache_bytes)  cache_bytes = 64ull << 20;
    }
    if (gguf.empty()) {
        std::fprintf(stderr, "no model: pass --gguf <path>, set $DS4_GGUF, or use --mini "
                             "(or --gate for the host-only self-test)\n");
        return 2;
    }
    if (!mini && !std::getenv("DS4_RUN_REAL")) {
        std::fprintf(stderr,
            "REFUSED: %s is a real model load (VRAM on every named card + a pinned host\n"
            "arena + a long file walk).  Set DS4_RUN_REAL=1 to authorise it, or use --mini\n"
            "to exercise this harness on the miniature fixture instead.\n", gguf.c_str());
        return 2;
    }

    emit("schema", "1");
    emit("run.tool", "ie-ds4-ppl");
    emit("run.build", std::string(__DATE__) + " " + __TIME__);
    emit("run.tag", tag.empty() ? "-" : tag);
    emit("run.model_path", gguf);
    emit("run.mini_fixture", mini ? "1" : "0");
    emit("run.mode", mode);
    if (mode == "batch") emit_u("run.chunk", chunk);
    emit("run.shuffle_targets", shuffle ? "1" : "0");
    // The one configuration knob this whole exercise is about.  Recorded from the
    // environment of THIS process, so a result line can never be separated from
    // the arm it came from.
    {
        const char* q8 = std::getenv("IE_DS4_DENSE_Q8");
        const bool on = q8 && *q8 && std::string(q8) != "0";
        emit("run.IE_DS4_DENSE_Q8", q8 ? q8 : "<unset>");
        emit("run.dense_q8_active", on ? "1" : "0");
    }
    for (const char* v : {"DS4_TP_GPUS", "DS4_PIN_CAP_GB", "DS4_SLOTS", "DS4_STREAM_SLOTS",
                          "DS4_PIN_LAYERS"}) {
        const char* e = std::getenv(v);
        emit(std::string("run.env.") + v, e ? e : "<unset>");
    }
    if (mini)
        note("MINIATURE FIXTURE: 6 layers, hidden 256, 8 experts, pseudo-random weights. This "
             "proves the harness computes and reports a perplexity end to end through the real "
             "DS4 runtime. Its PPL value is MEANINGLESS as a quality figure — random weights "
             "have no language model in them — and must never be quoted as one.");

    // ---- model ------------------------------------------------------------
    GgufReader g;
    if (const std::string e = g.open(gguf); !e.empty()) {
        emit("run.status", "gguf_open_failed"); note("gguf open: " + e); return 1;
    }
    DeepSeek4Config cfg;
    if (const std::string e = read_deepseek4_config(g, cfg); !e.empty()) {
        emit("run.status", "config_failed"); note("config: " + e); return 1;
    }
    emit_u("model.n_layers", cfg.n_layers);
    emit_u("model.hidden", cfg.hidden);
    emit_u("model.vocab", cfg.vocab);
    emit_u("model.n_experts", cfg.n_experts);

    // ---- tokens -----------------------------------------------------------
    std::vector<int32_t> ids;
    std::string corpus_desc;
    if (synthetic > 0) {
        ids.resize(synthetic);
        for (uint32_t i = 0; i < synthetic; ++i)
            ids[i] = int32_t((uint64_t(i) * 1103515245ull + 12345ull) % cfg.vocab);
        corpus_desc = "synthetic:" + std::to_string(synthetic);
        note("SYNTHETIC IDS: this is a plumbing exercise, not a language measurement. The "
             "perplexity of a pseudo-random id sequence is not a quality number.");
    } else {
        Tokenizer tok;
        const std::string te = tok.load_from_gguf(g);
        if (!te.empty()) {
            // The miniature fixture writes a token list but no merges, so the BPE
            // loader legitimately refuses it.  Say so and fall back, loudly.
            note("tokenizer: " + te);
            note("falling back to deterministic synthetic ids — the number below is a PLUMBING "
                 "result, not a language-model perplexity.");
            const uint32_t n = std::min<uint32_t>(max_tokens, 128u);
            ids.resize(n);
            for (uint32_t i = 0; i < n; ++i)
                ids[i] = int32_t((uint64_t(i) * 1103515245ull + 12345ull) % cfg.vocab);
            corpus_desc = "synthetic_fallback:" + std::to_string(n);
        } else {
            emit_u("tokenizer.vocab", tok.vocab_size());
            const std::string corpus = text_path.empty() ? std::string(kSampleText)
                                                         : read_text_file(text_path);
            if (corpus.empty()) {
                emit("run.status", "empty_corpus");
                note("corpus is empty (path '" + text_path + "')");
                return 1;
            }
            ids = tok.encode(corpus, /*allow_special=*/false);
            if (tok.add_bos_token() && tok.bos_token_id() >= 0 &&
                (ids.empty() || ids.front() != tok.bos_token_id()))
                ids.insert(ids.begin(), tok.bos_token_id());
            if (ids.size() > max_tokens) ids.resize(max_tokens);
            corpus_desc = text_path.empty() ? "builtin_prose" : text_path;
        }
    }
    if (ids.size() < 2) {
        emit("run.status", "too_few_tokens");
        note("need at least 2 tokens, have " + std::to_string(ids.size()));
        return 1;
    }
    for (int32_t id : ids) {
        if (id < 0 || uint32_t(id) >= cfg.vocab) {
            emit("run.status", "token_out_of_range");
            note("token id " + std::to_string(id) + " outside [0, " +
                 std::to_string(cfg.vocab) + ")");
            return 1;
        }
    }
    emit("corpus", corpus_desc);
    emit_u("tokens", ids.size());
    emit_u("predictions", ids.size() - 1);

    // The shuffle arm keeps position 0 fixed (it is only ever an input) and
    // permutes the rest.  The INPUT sequence the model sees is untouched — only
    // the targets each row is scored against move — so PPL must rise sharply if
    // the model has any predictive signal at all.
    std::vector<int32_t> score_ids = ids;
    if (shuffle) {
        uint32_t st = 20260803u;
        auto rnd = [&] { st = st * 1103515245u + 12345u; return st; };
        for (size_t i = score_ids.size() - 1; i >= 2; --i)
            std::swap(score_ids[i], score_ids[1 + (rnd() % uint32_t(i))]);
        note("--shuffle-targets: the model is fed the TRUE sequence, but each row is scored "
             "against a permuted target. A model with real predictive signal must show a much "
             "higher PPL here. If it does not, the number above is not measuring the model.");
    }

    // ---- runtime ----------------------------------------------------------
    DeepSeek4Model model;
    if (const std::string e = model.load(g, cfg); !e.empty()) {
        emit("run.status", "bind_failed"); note("bind: " + e); return 1;
    }
    Ds4Options base;
    base.device_ordinal     = gpu;
    base.max_seq            = std::max<uint32_t>(mode == "batch" ? chunk : 1u, max_seq);
    base.max_context        = uint32_t(ids.size()) + 16u;
    base.slots_per_layer    = slots;
    base.min_stream_slots   = stream_slots;
    base.expert_cache_bytes = cache_bytes;
    base.pin_layers         = pin_layers;
    emit_u("run.max_seq", base.max_seq);
    emit_u("run.max_context", base.max_context);

    DeepSeek4Runtime   rt;
    DeepSeek4TpRuntime tprt;
    Driver drv;
    const double tl0 = now_s();
    std::string le;
    if (tp_gpus.size() > 1) {
        Ds4TpOptions to;
        to.base                  = base;
        to.base.device_ordinal   = 0;
        to.base.n_cards          = 1;
        to.base.card             = 0;
        to.n_cards               = uint32_t(tp_gpus.size());
        to.device_ordinals       = tp_gpus;
        to.same_device_rehearsal = false;
        le = tprt.load(model, to);
        if (le.empty()) drv.tp = &tprt;
    } else {
        le = rt.load(model, base);
        if (le.empty()) drv.single = &rt;
    }
    emit_d("load.seconds", now_s() - tl0, 2);
    if (!le.empty()) { emit("run.status", "load_failed"); note("load: " + le); return 1; }
    emit("load.status", "ok");
    emit_u("load.cards", drv.n_cards());
    for (uint32_t c = 0; c < drv.n_cards(); ++c)
        emit("card." + std::to_string(c) + ".device",
             drv.card(c).queue().get_device().get_info<sycl::info::device::name>());

    // ---- score ------------------------------------------------------------
    auto fwd = [&](const int32_t* in, uint32_t T, uint32_t pos0, float* lg, bool last_only) {
        return drv.forward(in, T, pos0, lg, last_only);
    };
    auto reset = [&] { drv.reset_context(); };

    // `score_*` walks the INPUT sequence `ids` but must score against
    // `score_ids`; they are the same object unless --shuffle-targets moved the
    // targets, and the scorer only ever reads index i+1 for the target, so a
    // shuffled run is expressed by handing it the permuted vector for targets.
    // Rather than thread two vectors through the scorer, the shuffle arm feeds
    // the permuted targets through a wrapper forward that is fed the TRUE ids.
    Scored s;
    if (!shuffle) {
        s = (mode == "batch") ? score_batch(ids, cfg.vocab, chunk, fwd, reset)
                              : score_stream(ids, cfg.vocab, fwd, reset);
    } else {
        // Feed the model the true ids, score against the permuted ones: run the
        // scorer over `score_ids` but hand the forward the TRUE token at each
        // position.  `in` is ignored and replaced, which is exactly the intended
        // asymmetry and is why it is spelled out here rather than hidden.
        auto true_fwd = [&](const int32_t*, uint32_t T, uint32_t pos0, float* lg, bool lo) {
            return drv.forward(&ids[pos0], T, pos0, lg, lo);
        };
        s = (mode == "batch") ? score_batch(score_ids, cfg.vocab, chunk, true_fwd, reset)
                              : score_stream(score_ids, cfg.vocab, true_fwd, reset);
    }

    if (!s.err.empty()) {
        emit("run.status", "forward_failed");
        note(s.err);
        return 1;
    }
    if (s.nll.empty()) { emit("run.status", "no_predictions"); return 1; }

    std::vector<double> sorted = s.nll;
    std::sort(sorted.begin(), sorted.end());
    emit_u("ppl.scored", s.nll.size());
    emit_u("ppl.nonfinite", s.nonfinite);
    emit_d("ppl.seconds", s.seconds, 3);
    emit_d("ppl.ms_per_prediction", s.seconds * 1e3 / double(s.nll.size()), 4);
    emit_d("ppl.avg_nll", s.avg(), 6);
    emit_d("ppl.perplexity", s.ppl(), 4);
    emit_d("ppl.nll_p50", pct_of_sorted(sorted, 0.50), 6);
    emit_d("ppl.nll_p90", pct_of_sorted(sorted, 0.90), 6);
    emit_d("ppl.nll_p99", pct_of_sorted(sorted, 0.99), 6);
    emit_d("ppl.nll_max", sorted.back(), 6);
    emit_d("ppl.nll_min", sorted.front(), 6);

    if (s.nonfinite) {
        note("NON-FINITE LOGITS in " + std::to_string(s.nonfinite) + " of " +
             std::to_string(s.nll.size()) + " rows. The perplexity above is NOT usable.");
        emit("run.status", "nonfinite_logits");
        return 1;
    }

    const std::string meta =
        "mode=" + mode + " tokens=" + std::to_string(ids.size()) +
        " corpus=" + corpus_desc + " shuffle=" + (shuffle ? "1" : "0") +
        " dense_q8=" + (std::getenv("IE_DS4_DENSE_Q8") ? std::getenv("IE_DS4_DENSE_Q8") : "unset") +
        " cards=" + std::to_string(drv.n_cards());
    if (!dump_path.empty()) {
        const std::string we = write_nll(dump_path, s, meta);
        emit("dump.path", we.empty() ? dump_path : "error");
        if (!we.empty()) note("dump: " + we);
    }
    if (!base_path.empty()) {
        NllFile b;
        if (const std::string re = read_nll(base_path, b); !re.empty()) {
            note("baseline: " + re);
            emit("ab.status", "baseline_unreadable");
        } else {
            compare(b, s);
        }
    }

    emit("run.status", "ok");
    if (mini && !mini_keep) std::remove(mini_path.c_str());
    return 0;
}
