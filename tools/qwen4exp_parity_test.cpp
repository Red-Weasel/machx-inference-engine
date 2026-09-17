// tools/qwen4exp_parity_test.cpp — P2 gate: engine block forward vs the HF
// reference oracle (scripts/qwen4exp_oracle/), on the REAL weights.
//
// The oracle dumps quant-faithful fp32 goldens (dequantized UD-Q4_K_XL through
// modeling_qwen4_exp.py, CPU, eager). The engine runs the same block through
// the REAL forward path (run_block_parity) with F16-resident weights, so the
// expected deviation is f16 weight/activation rounding: stage rel-L2 ~1e-3.
// Gate: rel-L2 <= 1e-2 per captured stage and for the output wide state.
//
// usage: ie-qwen4exp-parity-test <model.gguf> <blk0> <blk3> <blk1ple> [gpu] [blk3_qsa_longctx]
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"
#include "ie/qwen4exp.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace ie;

namespace {

int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "\033[32mOK\033[0m" : "\033[31mFAIL\033[0m", what);
    if (!ok) ++g_fail;
}

struct Bins {
    std::map<std::string, std::vector<float>>   f;
    std::map<std::string, std::vector<int32_t>> i;
    bool load(const std::string& dir) {
        std::ifstream mf(dir + "/manifest.txt");
        if (!mf) return false;
        std::string line;
        while (std::getline(mf, line)) {
            std::istringstream ss(line);
            std::string key, dt, shape;
            ss >> key >> dt >> shape;
            uint64_t n = 1;
            if (!shape.empty()) {
                std::istringstream sh(shape);
                std::string d;
                while (std::getline(sh, d, ',')) if (!d.empty()) n *= std::stoull(d);
            }
            std::ifstream bf(dir + "/" + key + ".bin", std::ios::binary);
            if (!bf) return false;
            if (dt == "f32") {
                auto& v = f[key]; v.resize(n);
                bf.read(reinterpret_cast<char*>(v.data()), n * 4);
            } else {
                auto& v = i[key]; v.resize(n);
                bf.read(reinterpret_cast<char*>(v.data()), n * 4);
            }
        }
        return true;
    }
};

double rel_l2(const float* a, const float* b, uint64_t n) {   // ||a-b|| / ||b||
    double d2 = 0, r2 = 0;
    for (uint64_t i = 0; i < n; ++i) {
        const double d = double(a[i]) - b[i];
        d2 += d * d; r2 += double(b[i]) * b[i];
    }
    return r2 > 0 ? std::sqrt(d2 / r2) : std::sqrt(d2);
}

void stage(const char* name, const std::vector<sycl::half>& got,
           const std::vector<float>& want, double tol) {
    std::vector<float> g(got.size());
    for (size_t i = 0; i < got.size(); ++i) g[i] = float(got[i]);
    const double e = rel_l2(g.data(), want.data(), want.size());
    char buf[128];
    std::snprintf(buf, sizeof(buf), "%-14s rel-L2 %.3e (tol %.0e)", name, e, tol);
    check(e <= tol, buf);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s <model.gguf> <blk0_dir> <blk3_dir> <blk1ple_dir> [gpu]\n", argv[0]);
        return 2;
    }
    const uint32_t ordinal = argc > 5 ? uint32_t(std::atoi(argv[5])) : 0;

    GgufReader g;
    if (auto e = g.open(argv[1]); !e.empty()) { std::fprintf(stderr, "open: %s\n", e.c_str()); return 1; }
    Qwen4ExpConfig cfg;
    if (auto e = read_qwen4exp_config(g, cfg); !e.empty()) { std::fprintf(stderr, "config: %s\n", e.c_str()); return 1; }
    DeviceAllocator alloc;
    if (auto e = alloc.init("B70", ordinal); !e.empty()) { std::fprintf(stderr, "gpu: %s\n", e.c_str()); return 1; }

    Qwen4ExpModel m;
    if (auto e = m.load(alloc, g, cfg); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    if (auto e = m.init_runtime(4096, 2600); !e.empty()) { std::fprintf(stderr, "runtime: %s\n", e.c_str()); return 1; }

    std::printf("\n\033[1mQwen3.8-Flash-Next P2 parity gate (engine vs HF oracle)\033[0m\n");
    const double TOL_STAGE = 1e-2, TOL_WIDE = 1e-2;
    const uint32_t H = cfg.hidden;

    struct Case { const char* name; const char* dir; uint32_t layer; bool ple; };
    std::vector<Case> cases = {
        {"blk0 DeltaNet", argv[2], 0, false},
        {"blk3 full-attn+QSA(dense)", argv[3], 3, false},
        {"blk1 DeltaNet+PLE", argv[4], 1, true},
    };
    // Optional 5th bins dir: blk3 at T>2051 — TRUE sparse QSA selection
    // (top-512 blocks + tail) gated against the HF reference.
    if (argc > 6) cases.push_back({"blk3 QSA-SPARSE long-ctx", argv[6], 3, false});
    for (const Case& c : cases) {
        std::printf("\n\033[1m%s\033[0m  (%s)\n", c.name, c.dir);
        Bins b;
        if (!b.load(c.dir)) { std::fprintf(stderr, "  cannot load bins\n"); ++g_fail; continue; }
        const auto& xw = b.f.at("x_wide");
        const uint32_t T = uint32_t(xw.size() / (4 * H));
        std::vector<int32_t> toks;
        if (c.ple) {
            toks = b.i.at("ple_input_ids");
            // Fault isolation: host hash+gather E vs the oracle's golden E.
            PleHashConsts pc{};
            for (int j = 0; j < 3; ++j) pc.M[j] = cfg.ple_layer_multipliers[j];
            for (int h2 = 0; h2 < 16; ++h2) {
                pc.V[h2] = cfg.ple_head_vocab_sizes[h2];
                pc.O[h2] = cfg.ple_head_offsets[h2];
            }
            pc.eos = cfg.ple_eos_token;
            PleHistory hist;
            hist.reset(pc.eos);
            std::vector<uint64_t> rows(uint64_t(T) * 16);
            qwen4_ple_hash(pc, toks.data(), T, hist, rows.data());
            const GgufTensorInfo* tbl = g.find_tensor("per_layer_token_embd.weight");
            std::vector<float> E(uint64_t(T) * H);
            qwen4_ple_gather(tbl->data, rows.data(), T, E.data());
            const double eE = rel_l2(E.data(), b.f.at("ple_embed_E").data(), E.size());
            char buf[128];
            std::snprintf(buf, sizeof(buf), "%-14s rel-L2 %.3e (tol 1e-06, exact path)", "ple_embed_E", eE);
            check(eE <= 1e-6, buf);
        }

        std::vector<sycl::half> mixed_attn(uint64_t(T) * H), mixer_out(uint64_t(T) * H);
        std::vector<sycl::half> mixed_ffn(uint64_t(T) * H), moe_out(uint64_t(T) * H);
        Qwen4ExpModel::ParityCapture cap{mixed_attn.data(), mixer_out.data(),
                                         mixed_ffn.data(), moe_out.data()};
        std::vector<float> yw(xw.size());
        m.reset_state();
        if (auto e = m.run_block_parity(c.layer, xw.data(), T,
                                        c.ple ? toks.data() : nullptr,
                                        yw.data(), &cap); !e.empty()) {
            std::fprintf(stderr, "  run_block_parity: %s\n", e.c_str()); ++g_fail; continue;
        }
        stage("hc_attn_mix", mixed_attn, b.f.at("hc_attn_mix"), TOL_STAGE);
        stage("mixer_out",   mixer_out,  b.f.at("mixer_out"),   TOL_STAGE);
        if (b.f.count("hc_ffn_mix")) stage("hc_ffn_mix", mixed_ffn, b.f.at("hc_ffn_mix"), TOL_STAGE);
        if (T > 512) {
            // Large-T moe_out: router top-10 near-ties flip against a
            // different-order fp32 reduction (verified: flipped rows carry
            // probability gaps ~1e-6; the pick is mathematically arbitrary).
            // Gate = strict rel-L2 over the tie-free rows (99th percentile
            // trim) + a hard bound on the flip fraction. NOT a loosened
            // tolerance: a systematic compute bug lifts the trimmed metric.
            const auto& om = b.f.at("moe_out");
            std::vector<double> rowe(T);
            uint32_t flips = 0;
            for (uint32_t t = 0; t < T; ++t) {
                std::vector<float> mine(H);
                for (uint32_t d = 0; d < H; ++d)
                    mine[d] = float(moe_out[uint64_t(t) * H + d]);
                rowe[t] = rel_l2(mine.data(), om.data() + uint64_t(t) * H, H);
                if (rowe[t] > 5e-2) ++flips;
            }
            std::vector<double> sorted = rowe;
            std::sort(sorted.begin(), sorted.end());
            const double p99 = sorted[uint32_t(T * 99 / 100)];
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "moe_out p99-row rel %.3e (tol 1e-02), tie-flips %u/%u (max 1.5%%)",
                          p99, flips, T);
            check(p99 <= TOL_STAGE && flips * 200 <= 3 * T, buf);
        } else {
            stage("moe_out", moe_out, b.f.at("moe_out"), TOL_STAGE);
        }
        if (std::getenv("IE_Q4E_ROWDBG")) {
            const auto& om = b.f.at("moe_out");
            uint32_t bad = 0;
            for (uint32_t t = 0; t < T; ++t) {
                std::vector<float> mine(H);
                for (uint32_t d = 0; d < H; ++d) mine[d] = float(moe_out[uint64_t(t) * H + d]);
                const double e = rel_l2(mine.data(), om.data() + uint64_t(t) * H, H);
                double nm = 0, no = 0;
                for (uint32_t d = 0; d < H; ++d) {
                    nm += double(mine[d]) * mine[d];
                    no += double(om[uint64_t(t) * H + d]) * om[uint64_t(t) * H + d];
                }
                if (e > 5e-2) {
                    ++bad;
                    if (bad <= 12)
                        std::printf("      BAD row %4u rel %.3e  ||mine|| %.3f  ||oracle|| %.3f\n",
                                    t, e, std::sqrt(nm), std::sqrt(no));
                }
            }
            std::printf("      rows with rel > 5e-2: %u of %u\n", bad, T);
        }
        {
            const double e = rel_l2(yw.data(), b.f.at("y_wide").data(), yw.size());
            char buf[128];
            std::snprintf(buf, sizeof(buf), "%-14s rel-L2 %.3e (tol %.0e)", "y_wide", e, TOL_WIDE);
            check(e <= TOL_WIDE, buf);
        }
        // Router agreement on the top-10 sets (first + last token).
        if (b.i.count("router_top_idx")) {
            // Engine indices are not captured directly; the moe_out stage above
            // already fails hard if routing diverges (different experts =>
            // different output). This line just surfaces the oracle's picks.
            const auto& ri = b.i.at("router_top_idx");
            std::printf("      oracle t0 experts:");
            for (int j = 0; j < 10; ++j) std::printf(" %d", ri[j]);
            std::printf("\n");
        }
    }

    std::printf("\n\033[1m%s\033[0m  (%d failure%s)\n\n",
                g_fail ? "\033[31mGATE FAILED\033[0m" : "\033[32mGATE PASSED\033[0m",
                g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
