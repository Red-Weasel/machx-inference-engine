// tests/unit/deepseek4_attn_test.cpp — Phase 3 gate for the DeepSeek-V4
// attention stack (src/ops/deepseek4_attn.cpp, src/ops/deepseek4_cache.cpp).
//
// GROUND TRUTH
// ------------
// $DS4_PARITY2_DIR (default ~/.cache/ie-deepseek4-parity/parity2): 73 module
// invocations hooked out of a real shrunk DeepseekV4ForCausalLM forward
// (tools/deepseek4/parity_dump2.py).  Every shape is read from that dump's
// manifest.json, so regenerating with different --layers/--hidden does not
// silently invalidate the checks.
//
// A MISSING BLOB DIR IS A HARD FAILURE HERE, not a skip.  A skip that exits 0
// reads as a green ctest pass while testing nothing; the blobs live at a stable
// path and are regenerable with one command, which is printed on failure.
//
// WHAT THE DUMP DOES AND DOES NOT LET US GATE — stated up front so no claim is
// stronger than the evidence:
//   * parity_dump2.py records `named_parameters(recurse=False)`, so every
//     nn.Linear WEIGHT nested inside a hooked module is absent.  We therefore
//     cannot drive q_a_proj / q_b_proj / kv_proj / o_b_proj, nor the
//     compressors' kv_proj / gate_proj, nor the indexer's q_b_proj /
//     scorer.weights_proj, from hidden_states.
//   * What IS recorded is every RMSNorm's in/out, the compressors' and
//     indexer's in/out and position_bias, the scorer's in/out, the attention's
//     sinks / cos / sin / mask / position_ids, and o_a_proj's INPUT — which is
//     the attention output after the output-side RoPE.  That is enough to gate
//     the ENTIRE core attention path end to end, the compressors' RoPE +
//     running-entry bookkeeping, and the indexer's top-k, against real
//     reference numbers.
//   * The compressors' softmax POOLING (kv_proj/gate_proj output -> kv_norm
//     input) is NOT gated against reference blobs, because both its inputs are
//     unrecorded.  It is gated against a CPU double-precision transcription
//     instead, at BOTH shape families.  That catches kernel bugs but not a
//     mis-transcription; §5 says so explicitly rather than implying parity.
//
// TOLERANCES
// ----------
// Same model as the Phase 2 gate (tests/unit/deepseek4_parity_test.cpp):
// bound = C * u * sqrt(N) * scale with u = 2^-24, C = 4, N the reduction
// length, `scale` the reference tensor's max magnitude.  C is one fixed
// constant, applied everywhere, not tuned per check.  Every check prints
// observed-vs-bound so the margin is visible.  Discrete results (top-k indices,
// block-bias patterns, cache lengths) are compared EXACTLY — a mismatch there
// is a hard failure, not a tolerance question.
//
// NEGATIVE CONTROLS
// -----------------
// §7 re-runs the attention with (a) the output-side RoPE omitted and (b) its
// sign not conjugated, and FAILS if either still matches.  §1 does the same for
// the compress-theta table.  A bound with no negative control is a bound with
// no teeth.
#undef NDEBUG  // build is Release (-DNDEBUG); asserts must stay live here
#include "ie/deepseek4_attn.hpp"
#include "ie/deepseek4_cache.hpp"

#include "nlohmann/json.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr const char* G = "\033[32m";
constexpr const char* R = "\033[31m";
constexpr const char* Y = "\033[33m";
constexpr const char* Z = "\033[0m";

constexpr double kU = 5.9604644775390625e-8;  // fp32 unit roundoff
constexpr double kC = 4.0;                    // fixed safety constant

int g_fail = 0;

double dot_bound(size_t n, double scale) { return kC * kU * std::sqrt(double(n)) * scale; }

bool check(const char* name, double observed, double bound, size_t n) {
    const bool ok = observed <= bound && std::isfinite(observed);
    std::printf("  %s%-52s%s max|err|=%.3e  bound=%.3e (N=%zu)  %s%s%s\n",
                ok ? G : R, name, Z, observed, bound, n, ok ? G : R, ok ? "OK" : "FAIL", Z);
    if (!ok) ++g_fail;
    return ok;
}

bool check_exact(const char* name, long mismatches) {
    const bool ok = mismatches == 0;
    std::printf("  %s%-52s%s mismatches=%ld  %s%s%s\n",
                ok ? G : R, name, Z, mismatches, ok ? G : R, ok ? "OK" : "FAIL", Z);
    if (!ok) ++g_fail;
    return ok;
}

// A negative control PASSES when the mis-port's deviation is far outside the
// bound the corresponding POSITIVE check uses.  The floor is one uniform rule —
// kNegFactor times that bound — not a per-control number: the question a
// negative control answers is "would the positive check have caught this?", so
// the only principled reference scale is the positive check's own bound.
constexpr double kNegFactor = 100.0;

bool check_negative(const char* name, double observed, double positive_bound) {
    const double floor_ = kNegFactor * positive_bound;
    const bool ok = observed >= floor_ && std::isfinite(observed);
    std::printf("  %s%-52s%s deviation=%.3e  = %.0fx the %.3e bound  %s%s%s\n",
                ok ? G : R, name, Z, observed,
                positive_bound > 0.0 ? observed / positive_bound : 0.0,
                positive_bound, ok ? G : R, ok ? "OK" : "FAIL", Z);
    if (!ok) ++g_fail;
    return ok;
}

// ---------------------------------------------------------------------------
// Blob / manifest plumbing
// ---------------------------------------------------------------------------
using json = nlohmann::json;

bool file_exists(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

struct Blob {
    std::string file;
    std::vector<size_t> shape;
    size_t numel = 0;
};

struct Component {
    std::string name, cls;
    std::map<std::string, Blob> params, inputs, outputs;
};

std::string g_dir;
std::map<std::string, Component> g_comp;
json g_manifest;

Blob parse_blob(const json& j) {
    Blob b;
    b.file  = j.at("file").get<std::string>();
    b.numel = j.at("numel").get<size_t>();
    for (const auto& d : j.at("shape")) b.shape.push_back(d.get<size_t>());
    return b;
}

std::vector<float> load_blob(const Blob& b) {
    const std::string path = g_dir + "/" + b.file;
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    const size_t bytes = size_t(f.tellg());
    if (bytes != b.numel * sizeof(float)) {
        std::fprintf(stderr, "%s: %zu bytes, manifest says %zu\n",
                     path.c_str(), bytes, b.numel * sizeof(float));
        std::exit(1);
    }
    std::vector<float> v(b.numel);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), std::streamsize(bytes));
    return v;
}

const Component* find(const std::string& name) {
    auto it = g_comp.find(name);
    return it == g_comp.end() ? nullptr : &it->second;
}

const Component& require(const std::string& name) {
    const Component* c = find(name);
    if (!c) { std::fprintf(stderr, "manifest has no component '%s'\n", name.c_str()); std::exit(1); }
    return *c;
}

const Blob& slot(const Component& c, const char* sec, const std::string& key) {
    const std::map<std::string, Blob>* m =
        std::strcmp(sec, "params") == 0 ? &c.params
        : std::strcmp(sec, "inputs") == 0 ? &c.inputs : &c.outputs;
    auto it = m->find(key);
    if (it == m->end()) {
        std::fprintf(stderr, "component '%s' has no %s['%s']\n", c.name.c_str(), sec, key.c_str());
        std::exit(1);
    }
    return it->second;
}

std::vector<float> get(const std::string& comp, const char* sec, const std::string& key) {
    return load_blob(slot(require(comp), sec, key));
}

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------
sycl::queue make_queue() {
    sycl::device dev;
    bool found = false;
    for (const auto& d : sycl::device::get_devices())
        if (d.is_gpu()) { dev = d; found = true; break; }
    if (!found) { std::fprintf(stderr, "no GPU device\n"); std::exit(1); }
    return sycl::queue(dev, sycl::property_list{sycl::property::queue::in_order()});
}

template <typename T>
T* to_dev(sycl::queue& q, const std::vector<T>& h) {
    T* p = sycl::malloc_device<T>(std::max<size_t>(h.size(), 1), q);
    if (!h.empty()) q.memcpy(p, h.data(), h.size() * sizeof(T)).wait();
    return p;
}

template <typename T>
T* dev_alloc(sycl::queue& q, size_t n) {
    return sycl::malloc_device<T>(std::max<size_t>(n, 1), q);
}

template <typename T>
std::vector<T> from_dev(sycl::queue& q, const T* p, size_t n) {
    std::vector<T> h(n);
    if (n) q.memcpy(h.data(), p, n * sizeof(T)).wait();
    return h;
}

double max_abs(const std::vector<float>& v) {
    double m = 0.0;
    for (float x : v) m = std::max(m, double(std::fabs(x)));
    return m;
}

double max_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) m = std::max(m, double(std::fabs(a[i] - b[i])));
    return m;
}

// ---------------------------------------------------------------------------
// CPU references, transcribed from modeling_deepseek_v4.py, double accumulate
// ---------------------------------------------------------------------------

// DeepseekV4{HCA,CSA}Compressor / DeepseekV4Indexer window pooling
// (modeling:412-418 / 644-675 / 528-550).  `prior_*` may be empty.
std::vector<float> cpu_compress_pool(const std::vector<float>& chunk_kv,
                                     const std::vector<float>& chunk_gate,
                                     const std::vector<float>& bias,
                                     const std::vector<float>& prior_kv,
                                     const std::vector<float>& prior_gate,
                                     uint32_t n_win, uint32_t rate, uint32_t width,
                                     bool overlap) {
    const uint32_t src_w = overlap ? 2u * width : width;
    const uint32_t slots = overlap ? 2u * rate : rate;
    const double neg = -std::numeric_limits<double>::infinity();
    std::vector<float> out(size_t(n_win) * width, 0.f);
    std::vector<double> g(slots), v(slots);
    for (uint32_t w = 0; w < n_win; ++w) {
        for (uint32_t c = 0; c < width; ++c) {
            for (uint32_t j = 0; j < slots; ++j) {
                if (!overlap) {
                    g[j] = double(chunk_gate[(size_t(w) * rate + j) * src_w + c])
                         + double(bias[size_t(j) * src_w + c]);
                    v[j] = double(chunk_kv[(size_t(w) * rate + j) * src_w + c]);
                } else if (j < rate) {
                    if (w > 0) {
                        g[j] = double(chunk_gate[(size_t(w - 1) * rate + j) * src_w + c])
                             + double(bias[size_t(j) * src_w + c]);
                        v[j] = double(chunk_kv[(size_t(w - 1) * rate + j) * src_w + c]);
                    } else if (!prior_gate.empty()) {
                        // prior_gate is the RAW gate; bias row j is added here,
                        // exactly as for the w>0 case.  See the divergence note
                        // in include/ie/deepseek4_attn.hpp.
                        g[j] = double(prior_gate[size_t(j) * width + c])
                             + double(bias[size_t(j) * src_w + c]);
                        v[j] = double(prior_kv[size_t(j) * width + c]);
                    } else { g[j] = neg; v[j] = 0.0; }
                } else {
                    const uint32_t jj = j - rate;
                    g[j] = double(chunk_gate[(size_t(w) * rate + jj) * src_w + width + c])
                         + double(bias[size_t(jj) * src_w + width + c]);
                    v[j] = double(chunk_kv[(size_t(w) * rate + jj) * src_w + width + c]);
                }
            }
            double mx = neg;
            for (uint32_t j = 0; j < slots; ++j) mx = std::max(mx, g[j]);
            double den = 0.0, num = 0.0;
            for (uint32_t j = 0; j < slots; ++j) {
                const double e = (g[j] > neg) ? std::exp(g[j] - mx) : 0.0;
                den += e; num += e * v[j];
            }
            out[size_t(w) * width + c] = float(den > 0.0 ? num / den : 0.0);
        }
    }
    return out;
}

// DeepseekV4RMSNorm (modeling:46), double accumulate.
std::vector<float> cpu_rms_norm(const std::vector<float>& x, const std::vector<float>& w,
                                uint32_t rows, uint32_t n, double eps) {
    std::vector<float> y(x.size());
    for (uint32_t r = 0; r < rows; ++r) {
        double ss = 0.0;
        for (uint32_t j = 0; j < n; ++j) {
            const double v = double(x[size_t(r) * n + j]);
            ss += v * v;
        }
        const double inv = 1.0 / std::sqrt(ss / double(n) + eps);
        for (uint32_t j = 0; j < n; ++j)
            y[size_t(r) * n + j] = float(double(w[j]) * (double(x[size_t(r) * n + j]) * inv));
    }
    return y;
}

// apply_rotary_pos_emb (modeling:342) on the trailing rope slice, interleaved
// pairs, at the given absolute positions.  `inv` is the rope-type inv_freq.
void cpu_rope_apply(std::vector<float>& x, const std::vector<float>& inv,
                    const std::vector<int32_t>& positions,
                    uint32_t rows, uint32_t dim, uint32_t rope_dim) {
    const uint32_t nope = dim - rope_dim;
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t p = 0; p < rope_dim / 2u; ++p) {
            const double th = double(positions[r]) * double(inv[p]);
            const double c = std::cos(th), s = std::sin(th);
            const size_t i = size_t(r) * dim + nope + 2u * p;
            const double a = double(x[i]), b = double(x[i + 1]);
            x[i]     = float(a * c - b * s);
            x[i + 1] = float(b * c + a * s);
        }
}

// Solve a small symmetric positive-definite system by Gaussian elimination
// with partial pivoting (used by the indexer-scorer consistency check).
bool solve(std::vector<double>& A, std::vector<double>& b, uint32_t n) {
    for (uint32_t k = 0; k < n; ++k) {
        uint32_t piv = k;
        for (uint32_t r = k + 1; r < n; ++r)
            if (std::fabs(A[r * n + k]) > std::fabs(A[piv * n + k])) piv = r;
        if (std::fabs(A[piv * n + k]) < 1e-300) return false;
        if (piv != k) {
            for (uint32_t c = 0; c < n; ++c) std::swap(A[k * n + c], A[piv * n + c]);
            std::swap(b[k], b[piv]);
        }
        for (uint32_t r = k + 1; r < n; ++r) {
            const double f = A[r * n + k] / A[k * n + k];
            for (uint32_t c = k; c < n; ++c) A[r * n + c] -= f * A[k * n + c];
            b[r] -= f * b[k];
        }
    }
    for (uint32_t ri = n; ri-- > 0;) {
        double s = b[ri];
        for (uint32_t c = ri + 1; c < n; ++c) s -= A[ri * n + c] * b[c];
        b[ri] = s / A[ri * n + ri];
    }
    return true;
}

}  // namespace

// ===========================================================================
int main() {
    const char* env = std::getenv("DS4_PARITY2_DIR");
    if (env) {
        g_dir = env;
    } else {
        const char* home = std::getenv("HOME");
        g_dir = std::string(home ? home : ".") + "/.cache/ie-deepseek4-parity/parity2";
    }
    if (!file_exists(g_dir + "/manifest.json")) {
        std::fprintf(stderr,
            "%sdeepseek4_attn_test: FAIL — no parity blobs at %s%s\n"
            "  This test does NOT skip: a skip that exits 0 reads as a green pass.\n"
            "  Regenerate with:\n"
            "    python3 tools/deepseek4/parity_dump2.py --layers 4\n",
            R, g_dir.c_str(), Z);
        return 1;
    }
    {
        std::ifstream f(g_dir + "/manifest.json");
        f >> g_manifest;
        for (const auto& c : g_manifest.at("components")) {
            if (c.contains("error")) continue;
            Component comp;
            comp.name = c.at("component").get<std::string>();
            comp.cls  = c.contains("class") ? c.at("class").get<std::string>() : std::string();
            for (const auto& kv : c.at("params").items())  comp.params[kv.key()]  = parse_blob(kv.value());
            for (const auto& kv : c.at("inputs").items())  comp.inputs[kv.key()]  = parse_blob(kv.value());
            for (const auto& kv : c.at("outputs").items()) comp.outputs[kv.key()] = parse_blob(kv.value());
            g_comp[comp.name] = std::move(comp);
        }
    }

    sycl::queue q = make_queue();
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("blobs : %s\n", g_dir.c_str());

    // ---- shapes and constants, read from the dump rather than hardcoded ----
    const auto& tiny = g_manifest.at("tiny_config");
    const auto& pres = g_manifest.at("preserved_from_full");
    const uint32_t n_heads  = tiny.at("num_attention_heads").get<uint32_t>();
    const uint32_t o_groups = tiny.at("o_groups").get<uint32_t>();
    const uint32_t idx_topk = tiny.at("index_topk").get<uint32_t>();
    const uint32_t head_dim = pres.at("head_dim").get<uint32_t>();
    const uint32_t rope_dim = pres.at("qk_rope_head_dim").get<uint32_t>();
    const uint32_t idx_hd   = pres.at("index_head_dim").get<uint32_t>();
    const uint32_t window   = pres.at("sliding_window").get<uint32_t>();
    const float    rms_eps  = pres.at("rms_norm_eps").get<float>();
    const float    theta_main     = pres.at("rope_theta").get<float>();
    const float    theta_compress = pres.at("compress_rope_theta").get<float>();
    const std::vector<std::string> layer_types =
        g_manifest.at("tiny_layer_types").get<std::vector<std::string>>();
    const uint32_t half = rope_dim / 2u;
    const uint32_t T    = uint32_t(slot(require("model.layers.0.self_attn"), "inputs",
                                        "kw_position_ids").shape.back());
    std::printf("shapes: T=%u heads=%u head_dim=%u rope_dim=%u idx_hd=%u o_groups=%u "
                "topk=%u window=%u\n", T, n_heads, head_dim, rope_dim, idx_hd,
                o_groups, idx_topk, window);
    std::printf("types : ");
    for (const auto& s : layer_types) std::printf("%s ", s.c_str());
    std::printf("\n");

    // YaRN parameters of the "compress" table — ds4-hf/config.json rope_scaling.
    ie::Ds4RopeConfig rope_main;
    rope_main.theta = theta_main;
    ie::Ds4RopeConfig rope_comp;
    rope_comp.theta = theta_compress;
    rope_comp.yarn = true;
    rope_comp.factor = 16.f;
    rope_comp.beta_fast = 32.f;
    rope_comp.beta_slow = 1.f;
    rope_comp.original_max_pos = 65536u;
    rope_comp.attention_factor = 1.f;

    std::vector<int32_t> pos_seq(T);
    for (uint32_t i = 0; i < T; ++i) pos_seq[i] = int32_t(i);
    int32_t* d_pos_seq = to_dev(q, pos_seq);

    // =======================================================================
    std::puts("\n[1] Dual-theta RoPE tables (criterion 6)");
    // =======================================================================
    {
        const auto ref_main_c = get("model.layers.0.self_attn", "inputs", "kw_position_embeddings_main_0");
        const auto ref_main_s = get("model.layers.0.self_attn", "inputs", "kw_position_embeddings_main_1");
        const auto ref_cmp_c  = get("model.layers.0.self_attn", "inputs", "kw_position_embeddings_compress_0");
        const auto ref_cmp_s  = get("model.layers.0.self_attn", "inputs", "kw_position_embeddings_compress_1");

        auto build = [&](const ie::Ds4RopeConfig& cfg, std::vector<float>& c, std::vector<float>& s) {
            const auto inv = ie::ds4_rope_inv_freq(cfg, rope_dim);
            float* d_inv = to_dev(q, inv);
            float* d_c = dev_alloc<float>(q, size_t(T) * half);
            float* d_s = dev_alloc<float>(q, size_t(T) * half);
            ie::ds4_rope_cos_sin(q, d_inv, d_pos_seq, d_c, d_s, T, half,
                                 cfg.attention_factor).wait();
            c = from_dev(q, d_c, size_t(T) * half);
            s = from_dev(q, d_s, size_t(T) * half);
            sycl::free(d_inv, q); sycl::free(d_c, q); sycl::free(d_s, q);
        };
        std::vector<float> mc, ms, cc, cs;
        build(rope_main, mc, ms);
        build(rope_comp, cc, cs);
        // cos/sin are bounded by 1 and come from one multiply + a transcendental.
        const double b = dot_bound(1, 1.0) * 4.0;
        check("main cos (theta=10000, default)",  max_diff(mc, ref_main_c), b, 1);
        check("main sin (theta=10000, default)",  max_diff(ms, ref_main_s), b, 1);
        check("compress cos (theta=160000, yarn)", max_diff(cc, ref_cmp_c), b, 1);
        check("compress sin (theta=160000, yarn)", max_diff(cs, ref_cmp_s), b, 1);

        // Negative controls: the two tables must NOT be interchangeable, and the
        // yarn NTK-by-parts blend must actually be doing something.
        check_negative("NEG: main table used for compress branch",
                       max_diff(mc, ref_cmp_c), b);
        ie::Ds4RopeConfig no_yarn = rope_comp; no_yarn.yarn = false;
        std::vector<float> nc, ns;
        build(no_yarn, nc, ns);
        // NOTE on magnitude: the dump only covers positions 0..31, where YaRN's
        // 1/factor interpolation of the low-frequency dims has barely rotated
        // anything yet, so the ABSOLUTE cos deviation is only ~5e-4.  That is
        // still ~560x the parity bound, so the positive check above does
        // discriminate.  The position-independent statement is the one below:
        // yarn changes inv_freq itself by up to (1 - 1/factor) on the
        // interpolated dims.
        check_negative("NEG: compress theta without yarn scaling",
                       max_diff(nc, ref_cmp_c), b);
        {
            const auto iv_yarn = ie::ds4_rope_inv_freq(rope_comp, rope_dim);
            const auto iv_plain = ie::ds4_rope_inv_freq(no_yarn, rope_dim);
            double rel = 0.0;
            uint32_t touched = 0;
            for (uint32_t r = 0; r < half; ++r) {
                const double d = std::fabs(double(iv_yarn[r]) - double(iv_plain[r]))
                               / double(iv_plain[r]);
                if (d > 1e-9) ++touched;
                rel = std::max(rel, d);
            }
            const bool ok = rel >= 1.0 - 1.0 / double(rope_comp.factor) - 1e-3 && touched > 0;
            std::printf("  %s%-52s%s max rel diff=%.4f on %u/%u dims (expect %.4f)  %s%s%s\n",
                        ok ? G : R, "yarn NTK-by-parts actually rescales inv_freq", Z,
                        rel, touched, half, 1.0 - 1.0 / double(rope_comp.factor),
                        ok ? G : R, ok ? "OK" : "FAIL", Z);
            if (!ok) ++g_fail;
        }
    }

    // =======================================================================
    std::puts("\n[2] fp32 weighted RMSNorm vs reference blobs");
    // =======================================================================
    {
        auto gate_norm = [&](const char* comp, uint32_t n) {
            const Component* c = find(comp);
            if (!c) { std::printf("  %s(absent: %s)%s\n", Y, comp, Z); return; }
            const auto x   = load_blob(slot(*c, "inputs", "arg_0"));
            const auto w   = load_blob(slot(*c, "params", "weight"));
            const auto ref = load_blob(slot(*c, "outputs", "out"));
            const uint32_t rows = uint32_t(x.size() / n);
            float* dx = to_dev(q, x); float* dw = to_dev(q, w);
            float* dy = dev_alloc<float>(q, x.size());
            ie::ds4_rms_norm(q, dx, dw, dy, rows, n, rms_eps).wait();
            const auto y = from_dev(q, dy, x.size());
            check(comp + std::strlen("model.layers."), max_diff(y, ref),
                  dot_bound(n, max_abs(ref)), n);
            sycl::free(dx, q); sycl::free(dw, q); sycl::free(dy, q);
        };
        gate_norm("model.layers.2.self_attn.compressor.kv_norm", head_dim);
        gate_norm("model.layers.2.self_attn.compressor.indexer.kv_norm", idx_hd);
        gate_norm("model.layers.0.self_attn.kv_norm", head_dim);
    }

    // =======================================================================
    std::puts("\n[3] Compressor RoPE + running-entry bookkeeping (criterion 3)");
    // =======================================================================
    {
        // CSA layer: kv_norm.out --(compress rope at i*rate)--> compressor.out_0.
        // This gates the compress-theta selection, the emitted-entry positions
        // and the [B,1,T,head_dim] running list, at the ape (1024,4) family.
        const std::string base = "model.layers.2.self_attn.compressor";
        const auto kvn  = get(base + ".kv_norm", "outputs", "out");
        const auto ref  = get(base, "outputs", "out_0");
        const auto pb   = slot(require(base), "params", "position_bias");
        const uint32_t rate = uint32_t(pb.shape[0]);
        const uint32_t src_w = uint32_t(pb.shape[1]);
        const uint32_t n_win = uint32_t(kvn.size() / head_dim);
        std::printf("  CSA  compress_rate=%u position_bias=[%u,%u] (2*head_dim=%u) n_win=%u\n",
                    rate, rate, src_w, 2u * head_dim, n_win);
        if (src_w != 2u * head_dim) { std::printf("  %sCSA ape width unexpected%s\n", R, Z); ++g_fail; }

        const auto positions = ie::ds4_compress_positions(n_win, rate, 0);
        int32_t* d_p = to_dev(q, positions);
        const auto inv = ie::ds4_rope_inv_freq(rope_comp, rope_dim);
        float* d_inv = to_dev(q, inv);
        float* d_c = dev_alloc<float>(q, size_t(n_win) * half);
        float* d_s = dev_alloc<float>(q, size_t(n_win) * half);
        ie::ds4_rope_cos_sin(q, d_inv, d_p, d_c, d_s, n_win, half, 1.f).wait();
        float* d_x = to_dev(q, kvn);
        ie::ds4_rope_apply(q, d_x, d_c, d_s, d_x, n_win, 1, head_dim, rope_dim, +1.f).wait();
        const auto y = from_dev(q, d_x, kvn.size());
        check("CSA compressed entries (ape 1024x4)", max_diff(y, ref),
              dot_bound(2, max_abs(ref)), 2);
        sycl::free(d_p, q); sycl::free(d_inv, q); sycl::free(d_c, q);
        sycl::free(d_s, q); sycl::free(d_x, q);

        // HCA layer: the (512,128) family.  With T=32 < 128 the reference emits
        // ZERO windows, so what is gateable here is the window-alignment rule
        // itself: usable = (T / rate) * rate == 0, hence an empty entry list.
        const std::string hb = "model.layers.3.self_attn.compressor";
        const auto hpb = slot(require(hb), "params", "position_bias");
        const auto hout = slot(require(hb), "outputs", "out_0");
        const uint32_t hrate = uint32_t(hpb.shape[0]);
        std::printf("  HCA  compress_rate=%u position_bias=[%u,%zu] (head_dim=%u) "
                    "reference emitted %zu entries from T=%u\n",
                    hrate, hrate, hpb.shape[1], head_dim, hout.shape[2], T);
        if (hpb.shape[1] != head_dim) { std::printf("  %sHCA ape width unexpected%s\n", R, Z); ++g_fail; }
        {
            ie::Ds4LayerCache cache;
            ie::Ds4CacheConfig cfg;
            cfg.compress_ratio = hrate;
            cfg.head_dim = head_dim;
            cfg.index_head_dim = idx_hd;
            cfg.sliding_window = window;
            const std::string err = cache.init(q, cfg);
            if (!err.empty()) { std::printf("  %s%s%s\n", R, err.c_str(), Z); ++g_fail; }
            std::vector<float> zero(size_t(T) * head_dim, 0.f);
            float* dz = to_dev(q, zero);
            const auto ch = cache.store_compression_weights(ie::Ds4CacheEntry::Compressor, dz, dz, T);
            check_exact("HCA emits 0 windows from T<compress_rate",
                        long(ch.n_tokens) + long(hout.shape[2]));
            sycl::free(dz, q);
        }
    }

    // =======================================================================
    std::puts("\n[4] Compressor pooling, BOTH shape families (CPU reference)");
    // =======================================================================
    {
        // NOT a reference-blob gate: kv_proj / gate_proj outputs are not in the
        // dump (see the header note).  This compares the SYCL kernel against a
        // double-precision CPU transcription of modeling:644-675 / 412-418 at
        // the exact production shapes, including the Ca/Cb overlap and the
        // prior-window slice.  It catches kernel bugs, not transcription bugs.
        std::mt19937 rng(20260801);
        auto rnd = [&](size_t n) {
            std::vector<float> v(n);
            std::uniform_real_distribution<float> d(-2.f, 2.f);
            for (auto& x : v) x = d(rng);
            return v;
        };
        struct Case { const char* name; uint32_t n_win, rate, width; bool overlap, prior; };
        const Case cases[] = {
            {"HCA  rate=128 width=512 (no overlap)", 3, 128, head_dim, false, false},
            {"CSA  rate=4   width=512 (overlap, first call)", 5, 4, head_dim, true, false},
            {"CSA  rate=4   width=512 (overlap, prior slice)", 5, 4, head_dim, true, true},
            {"IDX  rate=4   width=128 (overlap, prior slice)", 5, 4, idx_hd, true, true},
        };
        for (const auto& cse : cases) {
            const uint32_t src_w = cse.overlap ? 2u * cse.width : cse.width;
            const auto kv   = rnd(size_t(cse.n_win) * cse.rate * src_w);
            const auto gate = rnd(size_t(cse.n_win) * cse.rate * src_w);
            const auto bias = rnd(size_t(cse.rate) * src_w);
            std::vector<float> pkv, pgate;
            if (cse.prior) { pkv = rnd(size_t(cse.rate) * cse.width); pgate = rnd(size_t(cse.rate) * cse.width); }
            const auto ref = cpu_compress_pool(kv, gate, bias, pkv, pgate,
                                               cse.n_win, cse.rate, cse.width, cse.overlap);
            float* dkv = to_dev(q, kv); float* dg = to_dev(q, gate); float* db = to_dev(q, bias);
            float* dpk = cse.prior ? to_dev(q, pkv) : nullptr;
            float* dpg = cse.prior ? to_dev(q, pgate) : nullptr;
            float* dout = dev_alloc<float>(q, size_t(cse.n_win) * cse.width);
            ie::ds4_compress_pool(q, dkv, dg, db, dpk, dpg, dout,
                                  cse.n_win, cse.rate, cse.width, cse.overlap).wait();
            const auto y = from_dev(q, dout, size_t(cse.n_win) * cse.width);
            const uint32_t slots = cse.overlap ? 2u * cse.rate : cse.rate;
            check(cse.name, max_diff(y, ref), dot_bound(slots, max_abs(ref)), slots);
            sycl::free(dkv, q); sycl::free(dg, q); sycl::free(db, q); sycl::free(dout, q);
            if (dpk) sycl::free(dpk, q);
            if (dpg) sycl::free(dpg, q);
        }
    }

    // =======================================================================
    std::puts("\n[4b] Whole compressor across MULTIPLE forward calls, both families");
    // =======================================================================
    {
        // The reference dump only ever runs T=32 tokens, so an HCA compressor
        // (compress_rate 128) never closes a window there — §3 can only check
        // that it correctly emits nothing.  This section drives the FULL
        // compressor pipeline (cache window buffering -> ds4_compress_pool ->
        // ds4_rms_norm -> compress-theta RoPE at i*rate + first_window_position
        // -> running entry list) across several ragged forward calls, at BOTH
        // shape families, against a double-precision CPU transcription of
        // modeling:394-428 (HCA) and modeling:623-686 (CSA).
        //
        // This is a self-reference check on the transcription, exactly like §4,
        // but it is the only thing that exercises rate=128 window straddling,
        // the buffer drain across calls, first_window_position advancing, and
        // the CSA overlap slice being carried between calls.
        std::mt19937 rng(987654321);
        auto rnd = [&](size_t n) {
            std::vector<float> v(n);
            std::uniform_real_distribution<float> d(-1.5f, 1.5f);
            for (auto& x : v) x = d(rng);
            return v;
        };
        struct Run { const char* name; uint32_t rate, width; bool overlap;
                     std::vector<uint32_t> calls; };
        const Run runs[] = {
            {"HCA rate=128 width=512", 128, head_dim, false, {50, 100, 90, 130, 200}},
            {"CSA rate=4   width=512",   4, head_dim, true,  {3, 5, 2, 10, 1, 7}},
            {"IDX rate=4   width=128",   4, idx_hd,   true,  {3, 5, 2, 10, 1, 7}},
        };
        const auto inv_comp = ie::ds4_rope_inv_freq(rope_comp, rope_dim);
        float* d_inv_comp = to_dev(q, inv_comp);
        for (const auto& rn : runs) {
            const uint32_t src_w = rn.overlap ? 2u * rn.width : rn.width;
            const auto norm_w = rnd(rn.width);
            const auto bias   = rnd(size_t(rn.rate) * src_w);

            ie::Ds4LayerCache cache;
            ie::Ds4CacheConfig cfg;
            cfg.compress_ratio = rn.rate;
            cfg.head_dim = rn.overlap && rn.width == idx_hd ? head_dim : rn.width;
            cfg.index_head_dim = rn.width;
            cfg.sliding_window = window;
            // The indexer entry of a CSA cache carries index_head_dim; the
            // compressor entry carries head_dim.  Pick the entry that matches
            // this run's width so the cache widths line up with the kernel's.
            const auto entry = (rn.width == head_dim) ? ie::Ds4CacheEntry::Compressor
                                                      : ie::Ds4CacheEntry::Indexer;
            const std::string e = cache.init(q, cfg);
            if (!e.empty()) { std::printf("  %s%s%s\n", R, e.c_str(), Z); ++g_fail; }

            float* d_bias = to_dev(q, bias);
            float* d_w    = to_dev(q, norm_w);

            // CPU mirror of the reference's state.
            std::vector<float> cpu_buf_kv, cpu_buf_gate, cpu_out;
            std::vector<float> cpu_ov_kv, cpu_ov_gate;
            uint32_t cpu_entries = 0;

            for (uint32_t ci = 0; ci < rn.calls.size(); ++ci) {
                const uint32_t n = rn.calls[ci];
                const auto kv   = rnd(size_t(n) * src_w);
                const auto gate = rnd(size_t(n) * src_w);

                // ---- CPU reference ----
                const uint32_t fwp = cpu_entries * rn.rate;
                std::vector<float> ckv = cpu_buf_kv, cgt = cpu_buf_gate;
                ckv.insert(ckv.end(), kv.begin(), kv.end());
                cgt.insert(cgt.end(), gate.begin(), gate.end());
                const uint32_t tot = uint32_t(ckv.size() / src_w);
                const uint32_t usable = (tot / rn.rate) * rn.rate;
                cpu_buf_kv.assign(ckv.begin() + size_t(usable) * src_w, ckv.end());
                cpu_buf_gate.assign(cgt.begin() + size_t(usable) * src_w, cgt.end());
                ckv.resize(size_t(usable) * src_w);
                cgt.resize(size_t(usable) * src_w);
                const uint32_t nw = usable / rn.rate;
                if (nw) {
                    auto pooled = cpu_compress_pool(ckv, cgt, bias, cpu_ov_kv, cpu_ov_gate,
                                                    nw, rn.rate, rn.width, rn.overlap);
                    if (rn.overlap) {
                        cpu_ov_kv.assign(size_t(rn.rate) * rn.width, 0.f);
                        cpu_ov_gate.assign(size_t(rn.rate) * rn.width, 0.f);
                        const size_t base = size_t(nw - 1) * rn.rate * src_w;
                        // RAW gate, matching Ds4LayerCache::update_overlap_state.
                        for (uint32_t j = 0; j < rn.rate; ++j)
                            for (uint32_t c = 0; c < rn.width; ++c) {
                                cpu_ov_kv[size_t(j) * rn.width + c] = ckv[base + size_t(j) * src_w + c];
                                cpu_ov_gate[size_t(j) * rn.width + c] = cgt[base + size_t(j) * src_w + c];
                            }
                    }
                    auto normed = cpu_rms_norm(pooled, norm_w, nw, rn.width, double(rms_eps));
                    const auto ps = ie::ds4_compress_positions(nw, rn.rate, fwp);
                    cpu_rope_apply(normed, inv_comp, ps, nw, rn.width, rope_dim);
                    cpu_out.insert(cpu_out.end(), normed.begin(), normed.end());
                    cpu_entries += nw;
                }

                // ---- engine ----
                float* d_kv = to_dev(q, kv);
                float* d_gt = to_dev(q, gate);
                const auto ch = cache.store_compression_weights(entry, d_kv, d_gt, n);
                if (ch.n_tokens != usable || ch.first_window_position != fwp) {
                    std::printf("  %s%s: cache chunk mismatch (%u vs %u, fwp %u vs %u)%s\n",
                                R, rn.name, ch.n_tokens, usable,
                                ch.first_window_position, fwp, Z);
                    ++g_fail;
                }
                const uint32_t enw = ch.n_tokens / rn.rate;
                if (enw) {
                    ie::Ds4OverlapSlice prior;
                    if (rn.overlap)
                        prior = cache.update_overlap_state(entry, ch.kv, ch.gate, enw);
                    float* d_pool = dev_alloc<float>(q, size_t(enw) * rn.width);
                    ie::ds4_compress_pool(q, ch.kv, ch.gate, d_bias,
                                          prior.valid ? prior.kv : nullptr,
                                          prior.valid ? prior.gate : nullptr,
                                          d_pool, enw, rn.rate, rn.width, rn.overlap).wait();
                    ie::ds4_rms_norm(q, d_pool, d_w, d_pool, enw, rn.width, rms_eps).wait();
                    const auto ps = ie::ds4_compress_positions(enw, rn.rate, ch.first_window_position);
                    int32_t* d_ps = to_dev(q, ps);
                    float* d_c = dev_alloc<float>(q, size_t(enw) * half);
                    float* d_s = dev_alloc<float>(q, size_t(enw) * half);
                    ie::ds4_rope_cos_sin(q, d_inv_comp, d_ps, d_c, d_s, enw, half, 1.f).wait();
                    ie::ds4_rope_apply(q, d_pool, d_c, d_s, d_pool, enw, 1,
                                       rn.width, rope_dim, +1.f).wait();
                    cache.update_compressor_states(entry, d_pool, enw, nullptr);
                    sycl::free(d_pool, q); sycl::free(d_ps, q);
                    sycl::free(d_c, q); sycl::free(d_s, q);
                }
                sycl::free(d_kv, q); sycl::free(d_gt, q);
            }

            const uint32_t got = cache.entry_count(entry);
            if (got != cpu_entries) {
                std::printf("  %s%s: entry_count %u vs %u%s\n", R, rn.name, got, cpu_entries, Z);
                ++g_fail;
            }
            // The cache stores its compressed rows fp16 (docs/deepseek4/73 Phase 2,
            // saturating cast at append); widen them back, and round the CPU
            // reference the same way so the bound below still measures the fp32
            // pooling arithmetic and not the storage width.
            std::vector<float> eng;
            if (cache.compressed_f16(entry)) {
                const auto eng16 = from_dev(q, static_cast<const sycl::half*>(cache.compressed(entry)),
                                            size_t(got) * rn.width);
                eng.assign(eng16.begin(), eng16.end());
                for (auto& v : cpu_out) v = float(sycl::half(std::fmin(std::fmax(v, -65504.f), 65504.f)));
            } else {
                eng = from_dev(q, static_cast<const float*>(cache.compressed(entry)), size_t(got) * rn.width);
            }
            std::printf("  %s: %u entries emitted over %zu calls\n",
                        rn.name, got, rn.calls.size());
            check(rn.name, max_diff(eng, cpu_out),
                  dot_bound(size_t(rn.width) + (rn.overlap ? 2u * rn.rate : rn.rate),
                            max_abs(cpu_out)),
                  rn.width);
            sycl::free(d_bias, q); sycl::free(d_w, q);
        }
        sycl::free(d_inv_comp, q);
    }

    // =======================================================================
    std::puts("\n[5] Lightning Indexer (criterion 2)");
    // =======================================================================
    {
        const std::string sc = "model.layers.2.self_attn.compressor.indexer.scorer";
        const std::string ix = "model.layers.2.self_attn.compressor.indexer";
        const auto qh   = get(sc, "inputs", "arg_0");   // [B,S,H,idx_hd], post-RoPE
        const auto keys = get(sc, "inputs", "arg_1");   // [B,T_keys,idx_hd]
        const auto sref = get(sc, "outputs", "out");    // [B,S,T_keys]
        const auto pids = get(ix, "inputs", "arg_2");   // [B,S] position_ids as f32
        const auto tref = get(ix, "outputs", "out");    // [B,S,top_k]
        const uint32_t idx_heads = uint32_t(slot(require(sc), "inputs", "arg_0").shape[2]);
        const uint32_t n_keys    = uint32_t(slot(require(sc), "inputs", "arg_1").shape[1]);
        const uint32_t top_k     = uint32_t(slot(require(ix), "outputs", "out").shape[2]);
        const uint32_t rate = uint32_t(slot(require("model.layers.2.self_attn.compressor"),
                                            "params", "position_bias").shape[0]);
        std::printf("  idx_heads=%u n_keys=%u top_k=%u compress_rate=%u\n",
                    idx_heads, n_keys, top_k, rate);

        // (a) scorer structure.  weights_proj.weight is NOT in the dump, so the
        //     per-token head weights w[t,:] are recovered by least squares from
        //     the n_keys equations the reference output gives us and the
        //     ReLU(q·k)*scale matrix this engine computes.  A small residual
        //     means the engine's A matrix spans the reference output; a wrong
        //     ReLU placement or a wrong scale factor would not.  This is a
        //     CONSISTENCY check, not full parity.
        {
            const double ss = 1.0 / std::sqrt(double(idx_hd));
            const double ws = 1.0 / std::sqrt(double(idx_heads));
            double worst = 0.0, ref_scale = 0.0;
            for (uint32_t t = 0; t < T; ++t) {
                std::vector<double> A(size_t(n_keys) * idx_heads);
                for (uint32_t e = 0; e < n_keys; ++e)
                    for (uint32_t h2 = 0; h2 < idx_heads; ++h2) {
                        double d = 0.0;
                        for (uint32_t d2 = 0; d2 < idx_hd; ++d2)
                            d += double(qh[(size_t(t) * idx_heads + h2) * idx_hd + d2])
                               * double(keys[size_t(e) * idx_hd + d2]);
                        A[size_t(e) * idx_heads + h2] = std::max(d, 0.0) * ss * ws;
                    }
                std::vector<double> N(size_t(idx_heads) * idx_heads, 0.0), rhs(idx_heads, 0.0);
                for (uint32_t a = 0; a < idx_heads; ++a) {
                    for (uint32_t b2 = 0; b2 < idx_heads; ++b2)
                        for (uint32_t e = 0; e < n_keys; ++e)
                            N[size_t(a) * idx_heads + b2] +=
                                A[size_t(e) * idx_heads + a] * A[size_t(e) * idx_heads + b2];
                    for (uint32_t e = 0; e < n_keys; ++e)
                        rhs[a] += A[size_t(e) * idx_heads + a] * double(sref[size_t(t) * n_keys + e]);
                }
                if (!solve(N, rhs, idx_heads)) continue;
                for (uint32_t e = 0; e < n_keys; ++e) {
                    double p = 0.0;
                    for (uint32_t h2 = 0; h2 < idx_heads; ++h2)
                        p += A[size_t(e) * idx_heads + h2] * rhs[h2];
                    worst = std::max(worst, std::fabs(p - double(sref[size_t(t) * n_keys + e])));
                    ref_scale = std::max(ref_scale, std::fabs(double(sref[size_t(t) * n_keys + e])));
                }
            }
            check("scorer ReLU/scale structure (LSQ residual)", worst,
                  dot_bound(size_t(idx_hd) * idx_heads, ref_scale), size_t(idx_hd) * idx_heads);

            // Same fit against an A matrix built WITHOUT the ReLU: if that also
            // fits, the check would be vacuous.  It must not.
            double worst_norelu = 0.0;
            for (uint32_t t = 0; t < T; ++t) {
                std::vector<double> A(size_t(n_keys) * idx_heads);
                for (uint32_t e = 0; e < n_keys; ++e)
                    for (uint32_t h2 = 0; h2 < idx_heads; ++h2) {
                        double d = 0.0;
                        for (uint32_t d2 = 0; d2 < idx_hd; ++d2)
                            d += double(qh[(size_t(t) * idx_heads + h2) * idx_hd + d2])
                               * double(keys[size_t(e) * idx_hd + d2]);
                        A[size_t(e) * idx_heads + h2] = d * ss * ws;
                    }
                std::vector<double> N(size_t(idx_heads) * idx_heads, 0.0), rhs(idx_heads, 0.0);
                for (uint32_t a = 0; a < idx_heads; ++a) {
                    for (uint32_t b2 = 0; b2 < idx_heads; ++b2)
                        for (uint32_t e = 0; e < n_keys; ++e)
                            N[size_t(a) * idx_heads + b2] +=
                                A[size_t(e) * idx_heads + a] * A[size_t(e) * idx_heads + b2];
                    for (uint32_t e = 0; e < n_keys; ++e)
                        rhs[a] += A[size_t(e) * idx_heads + a] * double(sref[size_t(t) * n_keys + e]);
                }
                if (!solve(N, rhs, idx_heads)) continue;
                for (uint32_t e = 0; e < n_keys; ++e) {
                    double p = 0.0;
                    for (uint32_t h2 = 0; h2 < idx_heads; ++h2)
                        p += A[size_t(e) * idx_heads + h2] * rhs[h2];
                    worst_norelu = std::max(worst_norelu,
                                            std::fabs(p - double(sref[size_t(t) * n_keys + e])));
                }
            }
            check_negative("NEG: same fit with the ReLU removed", worst_norelu,
                           dot_bound(size_t(idx_hd) * idx_heads, ref_scale));
        }

        // (b) the DISCRETE top-k selection, compared EXACTLY.
        std::vector<int32_t> pi(T);
        for (uint32_t t = 0; t < T; ++t) pi[t] = int32_t(std::lround(pids[t]));
        float*   d_sc = to_dev(q, sref);
        int32_t* d_pi = to_dev(q, pi);
        int32_t* d_tk = dev_alloc<int32_t>(q, size_t(T) * top_k);
        ie::ds4_indexer_topk(q, d_sc, d_pi, d_tk, T, n_keys, idx_topk, rate).wait();
        const auto tk = from_dev(q, d_tk, size_t(T) * top_k);
        long mism = 0;
        for (size_t i = 0; i < tk.size(); ++i)
            if (tk[i] != int32_t(std::lround(tref[i]))) ++mism;
        check_exact("indexer top-k indices EXACT (order + -1 sentinel)", mism);

        // (c) CSA block bias built from those indices, compared EXACTLY to the
        //     compressor's second return value.
        const auto bref = get("model.layers.2.self_attn.compressor", "outputs", "out_1");
        float* d_bb = dev_alloc<float>(q, size_t(T) * n_keys);
        ie::ds4_block_bias_topk(q, d_tk, d_bb, T, n_keys, top_k).wait();
        const auto bb = from_dev(q, d_bb, size_t(T) * n_keys);
        long bmism = 0;
        for (size_t i = 0; i < bb.size(); ++i) {
            const bool a = std::isinf(bb[i]) && bb[i] < 0.f;
            const bool b = std::isinf(bref[i]) && bref[i] < 0.f;
            if (a != b || (!a && bb[i] != bref[i])) ++bmism;
        }
        check_exact("CSA block_bias EXACT vs compressor.out_1", bmism);

        // (d) HCA dense block bias: causal threshold only (modeling:436-443).
        float* d_hb = dev_alloc<float>(q, size_t(T) * 7);
        ie::ds4_block_bias_dense(q, d_pi, d_hb, T, 7, rate).wait();
        const auto hb = from_dev(q, d_hb, size_t(T) * 7);
        long hmism = 0;
        for (uint32_t t = 0; t < T; ++t)
            for (uint32_t e = 0; e < 7; ++e) {
                const bool want_inf = int64_t(e) >= (int64_t(pi[t]) + 1) / int64_t(rate);
                const bool got_inf  = std::isinf(hb[size_t(t) * 7 + e]) && hb[size_t(t) * 7 + e] < 0.f;
                if (want_inf != got_inf || (!want_inf && hb[size_t(t) * 7 + e] != 0.f)) ++hmism;
            }
        check_exact("HCA dense block_bias causal threshold", hmism);

        sycl::free(d_sc, q); sycl::free(d_pi, q); sycl::free(d_tk, q);
        sycl::free(d_bb, q); sycl::free(d_hb, q);
    }

    // =======================================================================
    std::puts("\n[6] Sliding causal mask vs the model-level reference mask");
    // =======================================================================
    {
        const auto ref = get("model.layers.0.self_attn", "inputs", "kw_attention_mask");
        float* d_m = dev_alloc<float>(q, size_t(T) * T);
        ie::ds4_sliding_causal_mask(q, d_pos_seq, d_m, T, T, window).wait();
        const auto m = from_dev(q, d_m, size_t(T) * T);
        long mism = 0;
        for (size_t i = 0; i < m.size(); ++i) if (m[i] != ref[i]) ++mism;
        check_exact("sliding causal mask EXACT (window=128, T=32)", mism);
        sycl::free(d_m, q);
    }

    // =======================================================================
    std::puts("\n[7] Core attention, all three layer types (criteria 1 and 4)");
    // =======================================================================
    {
        const float scaling = 1.f / std::sqrt(float(head_dim));
        for (uint32_t li = 0; li < layer_types.size(); ++li) {
            const std::string p = "model.layers." + std::to_string(li) + ".self_attn";
            const bool compressed = layer_types[li] != "sliding_attention";
            const std::string rope_key = compressed ? "compress" : "main";

            const auto qn   = get(p + ".q_b_norm", "outputs", "out");      // [B,H,S,D]
            const auto kvn  = get(p + ".kv_norm", "outputs", "out");        // [B,S,D]
            const auto cos_ = get(p, "inputs", "kw_position_embeddings_" + rope_key + "_0");
            const auto sin_ = get(p, "inputs", "kw_position_embeddings_" + rope_key + "_1");
            const auto mask = get(p, "inputs", "kw_attention_mask");        // [B,1,S,S]
            const auto sink = get(p, "params", "sinks");
            const auto ref  = get(p + ".o_a_proj", "inputs", "arg_0");      // [B,S,g,H*D/g]

            // q comes out of the dump head-major; the engine is token-major.
            std::vector<float> qtok(qn.size());
            for (uint32_t hh = 0; hh < n_heads; ++hh)
                for (uint32_t t = 0; t < T; ++t)
                    std::memcpy(&qtok[(size_t(t) * n_heads + hh) * head_dim],
                                &qn[(size_t(hh) * T + t) * head_dim],
                                head_dim * sizeof(float));

            // Compressed entries (if any) are concatenated onto the KV axis and
            // the compressor's block_bias onto the mask (modeling:832, 840-844).
            std::vector<float> comp_kv, comp_bias;
            const Component* cc = find(p + ".compressor");
            if (cc) {
                comp_kv = load_blob(slot(*cc, "outputs", "out_0"));
                auto it = cc->outputs.find("out_1");
                if (it != cc->outputs.end()) comp_bias = load_blob(it->second);
            }
            const uint32_t n_comp = uint32_t(comp_kv.size() / head_dim);
            const uint32_t n_kv   = T + n_comp;
            if (n_comp && comp_bias.empty()) {
                std::printf("  %slayer %u: compressed entries with no block_bias%s\n", R, li, Z);
                ++g_fail;
            }

            float* d_q   = to_dev(q, qtok);
            float* d_kv  = dev_alloc<float>(q, size_t(n_kv) * head_dim);
            float* d_cos = to_dev(q, cos_);
            float* d_sin = to_dev(q, sin_);
            q.memcpy(d_kv, kvn.data(), kvn.size() * sizeof(float)).wait();
            if (n_comp)
                q.memcpy(d_kv + size_t(T) * head_dim, comp_kv.data(),
                         comp_kv.size() * sizeof(float)).wait();

            ie::ds4_rope_apply(q, d_q, d_cos, d_sin, d_q, T, n_heads, head_dim, rope_dim, +1.f).wait();
            ie::ds4_rope_apply(q, d_kv, d_cos, d_sin, d_kv, T, 1, head_dim, rope_dim, +1.f).wait();

            std::vector<float> fullmask(size_t(T) * n_kv);
            for (uint32_t t = 0; t < T; ++t) {
                std::memcpy(&fullmask[size_t(t) * n_kv], &mask[size_t(t) * T], T * sizeof(float));
                if (n_comp)
                    std::memcpy(&fullmask[size_t(t) * n_kv + T], &comp_bias[size_t(t) * n_comp],
                                n_comp * sizeof(float));
            }
            float* d_mask = to_dev(q, fullmask);
            float* d_sink = to_dev(q, sink);
            float* d_y    = dev_alloc<float>(q, size_t(T) * n_heads * head_dim);
            ie::ds4_attention(q, d_q, d_kv, d_mask, d_sink, d_y,
                              T, n_heads, head_dim, n_kv, scaling).wait();

            const auto pre = from_dev(q, d_y, size_t(T) * n_heads * head_dim);

            // The output-side conjugate rotation at position -i (modeling:868).
            ie::ds4_rope_apply(q, d_y, d_cos, d_sin, d_y, T, n_heads, head_dim, rope_dim, -1.f).wait();
            const auto post = from_dev(q, d_y, size_t(T) * n_heads * head_dim);

            const std::string tag = "L" + std::to_string(li) + " " + layer_types[li]
                                  + " (n_kv=" + std::to_string(n_kv) + ")";
            const double bnd = dot_bound(size_t(n_kv) * head_dim, max_abs(ref));
            check(tag.c_str(), max_diff(post, ref), bnd, size_t(n_kv) * head_dim);

            // Criterion 4 negative controls — both must deviate a lot.
            check_negative(("NEG L" + std::to_string(li) + ": output RoPE omitted").c_str(),
                           max_diff(pre, ref), bnd);
            {
                // `post` rotated by -sin a second time == the +sin (non-conjugate)
                // rotation applied to the raw attention output.
                float* d_w = to_dev(q, pre);
                ie::ds4_rope_apply(q, d_w, d_cos, d_sin, d_w, T, n_heads, head_dim, rope_dim, +1.f).wait();
                const auto wrong = from_dev(q, d_w, size_t(T) * n_heads * head_dim);
                check_negative(("NEG L" + std::to_string(li) + ": output RoPE sign not conjugated").c_str(),
                               max_diff(wrong, ref), bnd);
                sycl::free(d_w, q);
            }
            // The sink term.  IMPORTANT CAVEAT, stated rather than papered over:
            // the shrunk dump has NO checkpoint, and _init_weights zero-inits
            // `sinks` (modeling:1243), so every sink in these blobs is 0.0.
            // A sink of 0 is NOT a no-op — it adds exp(0 - m) to the softmax
            // denominator — so the control below proves the term is load-bearing
            // and correctly placed.  What it CANNOT discriminate is a per-head
            // INDEXING error (sinks[h] vs sinks[0]), because all four heads
            // carry the same value here.  That would need a dump with trained
            // or randomised sinks.
            {
                float* d_y2 = dev_alloc<float>(q, size_t(T) * n_heads * head_dim);
                ie::ds4_attention(q, d_q, d_kv, d_mask, nullptr, d_y2,
                                  T, n_heads, head_dim, n_kv, scaling).wait();
                ie::ds4_rope_apply(q, d_y2, d_cos, d_sin, d_y2, T, n_heads,
                                   head_dim, rope_dim, -1.f).wait();
                const auto nosink = from_dev(q, d_y2, size_t(T) * n_heads * head_dim);
                check_negative(("NEG L" + std::to_string(li) + ": sink logit dropped").c_str(),
                               max_diff(nosink, ref), bnd);
                sycl::free(d_y2, q);
            }

            sycl::free(d_q, q); sycl::free(d_kv, q); sycl::free(d_cos, q); sycl::free(d_sin, q);
            // The [T, H*D] -> [T, o_groups, H*D/o_groups] regroup the reference
            // does before o_a_proj is a pure reinterpretation of contiguous
            // memory, so `post` compares elementwise against o_a_proj's input.
            sycl::free(d_mask, q); sycl::free(d_sink, q); sycl::free(d_y, q);
        }
    }

    // =======================================================================
    std::puts("\n[8] Cache semantics and byte accounting (criterion 5)");
    // =======================================================================
    {
        // (a) sliding retention and the "returns the full concatenation" contract.
        ie::Ds4LayerCache c;
        ie::Ds4CacheConfig cfg;
        cfg.compress_ratio = 4;
        cfg.head_dim = head_dim;
        cfg.index_head_dim = idx_hd;
        cfg.sliding_window = window;
        const std::string err = c.init(q, cfg);
        if (!err.empty()) { std::printf("  %s%s%s\n", R, err.c_str(), Z); ++g_fail; }

        const uint32_t chunks[] = {1, 3, 5, 100, 100, 7};
        uint32_t fed = 0;
        long len_mism = 0, full_mism = 0, content_mism = 0;
        std::vector<float> expect;  // host mirror of every row ever appended
        for (uint32_t ci = 0; ci < 6; ++ci) {
            const uint32_t n = chunks[ci];
            std::vector<float> rows(size_t(n) * head_dim);
            for (uint32_t i = 0; i < n; ++i)
                for (uint32_t d = 0; d < head_dim; ++d)
                    rows[size_t(i) * head_dim + d] = float(fed + i) + 0.001f * float(d);
            expect.insert(expect.end(), rows.begin(), rows.end());
            float* dr = to_dev(q, rows);
            const uint32_t before = c.sliding_len();
            uint32_t full_len = 0;
            const float* dfull = c.update_sliding(dr, n, &full_len);
            if (full_len != before + n) ++full_mism;
            const auto full = from_dev(q, dfull, size_t(full_len) * head_dim);
            fed += n;
            const uint32_t want_keep = std::min(fed, window - 1u);
            if (c.sliding_len() != want_keep) ++len_mism;
            // The retained rows must be the LAST `want_keep` rows ever fed.
            for (uint32_t i = 0; i < want_keep; ++i) {
                const size_t src = size_t(fed - want_keep + i) * head_dim;
                const size_t dst = size_t(full_len - want_keep + i) * head_dim;
                if (std::fabs(full[dst] - expect[src]) > 0.f) ++content_mism;
            }
            sycl::free(dr, q);
        }
        std::printf("  fed=%u sliding_len=%u (window-1=%u) cumulative=%llu\n",
                    fed, c.sliding_len(), window - 1u,
                    (unsigned long long)c.cumulative_length());
        check_exact("update_sliding returns retained+T rows", full_mism);
        check_exact("sliding retention == min(t, sliding_window-1)", len_mism);
        check_exact("retained rows are the LAST window-1 fed", content_mism);
        check_exact("cumulative_length tracks every fed token",
                    long(c.cumulative_length()) - long(fed));

        // (b) window buffering: only whole windows are peeled off, the rest is
        //     retained, and first_window_position advances by rate per entry.
        ie::Ds4LayerCache c2;
        c2.init(q, cfg);
        long buf_mism = 0, fwp_mism = 0;
        uint32_t total_tok = 0, emitted = 0;
        const uint32_t feed[] = {3, 3, 2, 9, 1};
        for (uint32_t ci = 0; ci < 5; ++ci) {
            const uint32_t n = feed[ci];
            std::vector<float> rows(size_t(n) * 2u * head_dim, 1.f);
            float* dr = to_dev(q, rows);
            const auto ch = c2.store_compression_weights(ie::Ds4CacheEntry::Compressor, dr, dr, n);
            total_tok += n;
            if (ch.first_window_position != emitted * 4u) ++fwp_mism;
            const uint32_t want = ((total_tok - (total_tok % 4u)) - emitted * 4u);
            if (ch.n_tokens != want) ++buf_mism;
            const uint32_t n_win = ch.n_tokens / 4u;
            std::vector<float> comp(size_t(n_win) * head_dim, 0.f);
            float* dc = to_dev(q, comp);
            uint32_t cnt = 0;
            c2.update_compressor_states(ie::Ds4CacheEntry::Compressor, dc, n_win, &cnt);
            emitted += n_win;
            if (cnt != emitted) ++fwp_mism;
            sycl::free(dr, q); sycl::free(dc, q);
        }
        check_exact("store_compression_weights peels whole windows only", buf_mism);
        check_exact("first_window_position == entry_count * compress_rate", fwp_mism);

        // (c) byte accounting: the whole 43-layer schedule, measured between two
        //     window-aligned context lengths so every non-growing term cancels.
        const std::vector<uint32_t> ratios = [] {
            std::vector<uint32_t> r{0, 0};
            for (uint32_t l = 2; l < 43; ++l) r.push_back(l % 2 == 0 ? 4u : 128u);
            return r;
        }();
        std::vector<ie::Ds4LayerCache> layers(ratios.size());
        for (size_t l = 0; l < ratios.size(); ++l) {
            ie::Ds4CacheConfig lc;
            lc.compress_ratio = ratios[l];
            lc.head_dim = head_dim;
            lc.index_head_dim = idx_hd;
            lc.sliding_window = window;
            const std::string e = layers[l].init(q, lc);
            if (!e.empty()) { std::printf("  %s%s%s\n", R, e.c_str(), Z); ++g_fail; }
        }
        const uint32_t step = 128, n1 = 1024, n2 = 2048;
        std::vector<float> sl(size_t(step) * head_dim, 0.5f);
        std::vector<float> src(size_t(step) * 2u * head_dim, 0.25f);
        std::vector<float> emitbuf(size_t(step) * head_dim, 0.125f);
        float* d_sl = to_dev(q, sl);
        float* d_src = to_dev(q, src);
        float* d_emit = to_dev(q, emitbuf);
        uint64_t bytes_at_n1 = 0;
        for (uint32_t done = 0; done < n2; done += step) {
            for (size_t l = 0; l < ratios.size(); ++l) {
                auto& lay = layers[l];
                lay.update_sliding(d_sl, step, nullptr);
                if (!lay.has_compressor()) continue;
                for (uint32_t ei = 0; ei < (lay.has_indexer() ? 2u : 1u); ++ei) {
                    const auto e = ie::Ds4CacheEntry(ei);
                    const auto ch = lay.store_compression_weights(e, d_src, d_src, step);
                    const uint32_t nw = ch.n_tokens / ratios[l];
                    if (nw) lay.update_compressor_states(e, d_emit, nw, nullptr);
                }
            }
            if (done + step == n1) {
                for (const auto& lay : layers) bytes_at_n1 += lay.state_bytes(2);
            }
        }
        uint64_t bytes_at_n2 = 0, alloc = 0;
        for (const auto& lay : layers) { bytes_at_n2 += lay.state_bytes(2); alloc += lay.allocated_bytes(); }
        const double per_tok = double(bytes_at_n2 - bytes_at_n1) / double(n2 - n1);
        std::printf("  state_bytes(2) @%u = %llu, @%u = %llu  -> %.2f B/token\n",
                    n1, (unsigned long long)bytes_at_n1, n2,
                    (unsigned long long)bytes_at_n2, per_tok);
        std::printf("  fp32 device capacity actually held by the 43 caches: %.1f MiB "
                    "(includes the window scratch and geometric-growth slack)\n",
                    double(alloc) / 1048576.0);
        const bool ok = std::fabs(per_tok - 6880.0) <= 6880.0 * 0.05;
        std::printf("  %s%-52s%s %.2f B/token vs 6880 +/-5%%  %s%s%s\n",
                    ok ? G : R, "KV growth per token", Z, per_tok,
                    ok ? G : R, ok ? "OK" : "FAIL", Z);
        if (!ok) ++g_fail;

        // The 128-position sliding retention, verified on the real schedule.
        long ret = 0;
        for (const auto& lay : layers) if (lay.sliding_len() != window - 1u) ++ret;
        check_exact("all 43 layers retain exactly sliding_window-1 rows", ret);

        sycl::free(d_sl, q); sycl::free(d_src, q); sycl::free(d_emit, q);
    }

    sycl::free(d_pos_seq, q);

    std::printf("\n%s%s%s\n", g_fail ? R : G,
                g_fail ? "deepseek4_attn_test: FAILED" : "deepseek4_attn_test: ALL CHECKS PASSED", Z);
    return g_fail ? 1 : 0;
}
