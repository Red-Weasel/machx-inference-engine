// tools/ie_validate_chunking.cpp
//
// Deterministic chunking-equivalence validator.  Runs the same input
// twice — once as a single forward(T=1024) call, once as 4 chunks of
// forward(T=256) — and compares per-layer activations at the last
// position (which should be IDENTICAL for a correct engine).
//
// This tool exists because of the prefill-PPL anomaly observed
// 2026-05-02:  ie-perplexity --prefill-chunk N --max-tokens N+50
// shows catastrophic PPL spikes when N >= 512:
//   prefill=256, +50 streamed: PPL  4.94
//   prefill=512, +50 streamed: PPL 17.80
//   prefill=768, +50 streamed: PPL 10.80
//   prefill=1024,+50 streamed: PPL 65.05
// Same model, same corpus, same kernels — only the prefill batch size
// differs.  Production paths chunk at T=256 so the bug was hidden.
//
// Methodology:
//   1. Tokenize a corpus, take first 1024 tokens.
//   2. Path A: forward(T=1024, start_pos=0) with dump_prefix="A_dumps/"
//   3. Reset KV + DN.
//   4. Path B: forward(T=256) × 4 with dump_prefix="B_dumps/c{0..3}"
//      at start_pos = 0, 256, 512, 768.
//   5. For each layer (L00 = embed, L01..L40 = post-layer residual,
//      L41 = final norm), compare:
//        A's row 1023 (last position of single call)
//        B's chunk-3 row 255 (last position of last chunk = absolute 1023)
//   6. Report max abs diff per layer; first layer with significant
//      divergence is where the bug lives.

#include "ie/allocator.hpp"
#include "ie/deltanet_state.hpp"
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/qwen36.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <string>

// IE_VALIDATE_C_ONLY: when 1, the validator skips Paths A, B, A2 (and
// their compares) and runs only Path C1 + Path C2.  Useful when only the
// streaming-T=1 determinism check is wanted.
#ifndef IE_VALIDATE_C_ONLY
#define IE_VALIDATE_C_ONLY 0
#endif
#include <vector>

namespace {

constexpr uint32_t T_TOTAL = 1024;
constexpr uint32_t CHUNK   = 256;
constexpr uint32_t N_CHUNKS = T_TOTAL / CHUNK;     // 4
constexpr int      N_DUMP_LAYERS = 42;             // L00..L41

// Fast 64-bit hash for bit-equality detection (FNV-1a on 64-bit words).
// Not cryptographic; only used to compare two large byte buffers for
// any-bit-difference.  Step 12 of the bisect (2026-05-03).
uint64_t hash64(const void* data, size_t bytes,
                uint64_t seed = 0xcbf29ce484222325ULL) {
    uint64_t h = seed;
    const uint8_t* p = static_cast<const uint8_t*>(data);
    const size_t nwords = bytes / 8;
    const uint64_t* w = reinterpret_cast<const uint64_t*>(p);
    for (size_t i = 0; i < nwords; ++i) {
        h ^= w[i]; h *= 0x100000001b3ULL;
    }
    for (size_t i = nwords * 8; i < bytes; ++i) {
        h ^= uint64_t(p[i]); h *= 0x100000001b3ULL;
    }
    return h;
}

constexpr uint32_t MAX_KV_LAYERS = 16;
constexpr uint32_t MAX_DN_LAYERS = 32;
struct StateHashes {
    uint32_t n_layers_full = 0;
    uint32_t n_layers_dn   = 0;
    uint32_t lengths[MAX_KV_LAYERS]{};
    uint64_t k_per_layer[MAX_KV_LAYERS]{};
    uint64_t v_per_layer[MAX_KV_LAYERS]{};
    uint64_t dn_state_per_layer[MAX_DN_LAYERS]{};
    uint64_t dn_conv_per_layer [MAX_DN_LAYERS]{};
    uint64_t combined() const {
        uint64_t h = 0xcbf29ce484222325ULL;
        for (uint32_t L = 0; L < n_layers_full; ++L) {
            h ^= k_per_layer[L]; h *= 0x100000001b3ULL;
            h ^= v_per_layer[L]; h *= 0x100000001b3ULL;
            h ^= uint64_t(lengths[L]); h *= 0x100000001b3ULL;
        }
        for (uint32_t L = 0; L < n_layers_dn; ++L) {
            h ^= dn_state_per_layer[L]; h *= 0x100000001b3ULL;
            h ^= dn_conv_per_layer [L]; h *= 0x100000001b3ULL;
        }
        return h;
    }
};

// Capture a hash of the populated portion of K/V cache + entire DN state.
// Reads only `length(layer) * head_dim * sizeof(half)` per kv-head per layer
// to avoid hashing uninitialised cache slots (which differ run-to-run by
// design — only `kv.reset()` clears lengths, not memory).
StateHashes capture_state(sycl::queue& q,
                          const ie::KvCache& kv,
                          const ie::DeltaNetState& dn,
                          std::vector<uint8_t>& host_buf) {
    StateHashes h{};
    const auto& kvc = kv.config();
    h.n_layers_full = std::min(uint32_t(MAX_KV_LAYERS), kvc.n_layers_full);
    const uint64_t per_kvhead_elems = uint64_t(kvc.max_ctx) * kvc.head_dim;
    const uint64_t per_layer_elems  = uint64_t(kvc.n_kv_heads) * per_kvhead_elems;
    const sycl::half* k_base = kv.k_ptr();
    const sycl::half* v_base = kv.v_ptr();
    for (uint32_t L = 0; L < h.n_layers_full; ++L) {
        const uint32_t len = kv.length(L);
        h.lengths[L] = len;
        if (len == 0) {
            h.k_per_layer[L] = 0xcbf29ce484222325ULL;
            h.v_per_layer[L] = 0xcbf29ce484222325ULL;
            continue;
        }
        const size_t bytes_per_kvh = size_t(len) * kvc.head_dim * sizeof(sycl::half);
        if (host_buf.size() < bytes_per_kvh) host_buf.resize(bytes_per_kvh);
        uint64_t k_h = 0xcbf29ce484222325ULL;
        uint64_t v_h = 0xcbf29ce484222325ULL;
        for (uint32_t kvh = 0; kvh < kvc.n_kv_heads; ++kvh) {
            const sycl::half* k_src = k_base + L * per_layer_elems + kvh * per_kvhead_elems;
            const sycl::half* v_src = v_base + L * per_layer_elems + kvh * per_kvhead_elems;
            q.memcpy(host_buf.data(), k_src, bytes_per_kvh).wait();
            k_h = hash64(host_buf.data(), bytes_per_kvh, k_h);
            q.memcpy(host_buf.data(), v_src, bytes_per_kvh).wait();
            v_h = hash64(host_buf.data(), bytes_per_kvh, v_h);
        }
        h.k_per_layer[L] = k_h;
        h.v_per_layer[L] = v_h;
    }
    const auto& dnc = dn.config();
    h.n_layers_dn = std::min(uint32_t(MAX_DN_LAYERS), dnc.n_layers_linear);
    const size_t dn_state_per_layer_bytes =
        size_t(dnc.n_v_heads) * dnc.v_head_dim * dnc.k_head_dim * sizeof(float);
    const size_t dn_conv_per_layer_bytes  =
        size_t(dnc.conv_channels) * (dnc.conv_kernel - 1) * sizeof(sycl::half);
    const size_t dn_state_total = dn_state_per_layer_bytes * dnc.n_layers_linear;
    const size_t dn_conv_total  = dn_conv_per_layer_bytes  * dnc.n_layers_linear;
    if (host_buf.size() < std::max(dn_state_total, dn_conv_total))
        host_buf.resize(std::max(dn_state_total, dn_conv_total));

    // One D2H of the entire DN state, then per-layer hash on the host slab.
    q.memcpy(host_buf.data(), dn.state_ptr(), dn_state_total).wait();
    for (uint32_t L = 0; L < h.n_layers_dn; ++L) {
        h.dn_state_per_layer[L] = hash64(
            host_buf.data() + L * dn_state_per_layer_bytes,
            dn_state_per_layer_bytes);
    }
    q.memcpy(host_buf.data(), dn.conv_state_ptr(), dn_conv_total).wait();
    for (uint32_t L = 0; L < h.n_layers_dn; ++L) {
        h.dn_conv_per_layer[L] = hash64(
            host_buf.data() + L * dn_conv_per_layer_bytes,
            dn_conv_per_layer_bytes);
    }
    return h;
}

// Read one row of `H` fp32 values from a dump file.
std::vector<float> read_row(const std::string& path, uint32_t row, uint32_t H) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "  open fail: %s\n", path.c_str());
        return {};
    }
    f.seekg(uint64_t(row) * H * sizeof(float));
    std::vector<float> v(H);
    f.read(reinterpret_cast<char*>(v.data()), uint64_t(H) * sizeof(float));
    if (!f) {
        std::fprintf(stderr, "  read fail: %s row=%u H=%u\n", path.c_str(), row, H);
        return {};
    }
    return v;
}

struct LayerCmp {
    double max_abs_diff;
    double max_rel_diff;
    double l2_diff;
    double a_l2;
    double b_l2;
    int    n_nan_a;
    int    n_nan_b;
    int    n_inf_a;
    int    n_inf_b;
};

LayerCmp compare_rows(const std::vector<float>& a, const std::vector<float>& b) {
    LayerCmp c{};
    if (a.size() != b.size() || a.empty()) {
        c.max_abs_diff = -1; return c;
    }
    double sum_d2 = 0, sum_a2 = 0, sum_b2 = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::isnan(a[i])) ++c.n_nan_a;
        if (std::isnan(b[i])) ++c.n_nan_b;
        if (std::isinf(a[i])) ++c.n_inf_a;
        if (std::isinf(b[i])) ++c.n_inf_b;
        const double da = double(a[i]);
        const double db = double(b[i]);
        const double diff = da - db;
        const double absd = std::fabs(diff);
        const double maxabs = std::max(std::fabs(da), std::fabs(db));
        const double rel = (maxabs > 1e-12) ? absd / maxabs : 0.0;
        if (absd > c.max_abs_diff) c.max_abs_diff = absd;
        if (rel  > c.max_rel_diff) c.max_rel_diff = rel;
        sum_d2 += diff * diff;
        sum_a2 += da * da;
        sum_b2 += db * db;
    }
    c.l2_diff = std::sqrt(sum_d2);
    c.a_l2    = std::sqrt(sum_a2);
    c.b_l2    = std::sqrt(sum_b2);
    return c;
}

}  // namespace

int main(int argc, char** argv) {
    std::string gguf_path =
        "/home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf";
    std::string text_path = "/tmp/ppl_test.txt";
    std::string dump_root = "/tmp/ie_validate_chunking";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--gguf"      && i + 1 < argc) gguf_path = argv[++i];
        else if (a == "--text"      && i + 1 < argc) text_path = argv[++i];
        else if (a == "--dump-root" && i + 1 < argc) dump_root = argv[++i];
    }

    std::printf("ie-validate-chunking — single T=%u vs %u×T=%u logit-bisect\n",
                T_TOTAL, N_CHUNKS, CHUNK);
    std::printf("  gguf      : %s\n", gguf_path.c_str());
    std::printf("  text      : %s\n", text_path.c_str());
    std::printf("  dump_root : %s\n", dump_root.c_str());

    // 1. Load GGUF + alloc + tokenizer
    ie::GgufReader g;
    if (auto err = g.open(gguf_path); !err.empty()) {
        std::fprintf(stderr, "gguf open: %s\n", err.c_str()); return 1;
    }
    ie::DeviceAllocator alloc;
    if (auto err = alloc.init(); !err.empty()) {
        std::fprintf(stderr, "alloc: %s\n", err.c_str()); return 1;
    }
    auto& q = alloc.queue();
    std::printf("  device    : %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());

    ie::Tokenizer tok;
    if (auto err = tok.load_from_gguf(g); !err.empty()) {
        std::fprintf(stderr, "tokenizer: %s\n", err.c_str()); return 1;
    }

    // 2. Model.
    std::printf("  loading model...\n");
    ie::QwenConfig cfg;
    ie::QwenModel  model;
    if (auto err = model.load(alloc, g, cfg); !err.empty()) {
        std::fprintf(stderr, "model.load: %s\n", err.c_str()); return 1;
    }

    // 3. Tokenize and slice.
    std::ifstream tf(text_path);
    if (!tf) {
        std::fprintf(stderr, "open text: %s\n", text_path.c_str()); return 1;
    }
    std::string corpus((std::istreambuf_iterator<char>(tf)),
                        std::istreambuf_iterator<char>());
    auto ids = tok.encode(corpus, /*allow_special=*/false);
    if (tok.add_bos_token() && tok.bos_token_id() >= 0 &&
        (ids.empty() || ids.front() != tok.bos_token_id())) {
        ids.insert(ids.begin(), tok.bos_token_id());
    }
    if (ids.size() < T_TOTAL) {
        std::fprintf(stderr, "corpus too short: %zu < %u\n", ids.size(), T_TOTAL);
        return 1;
    }
    ids.resize(T_TOTAL);
    std::printf("  tokenized: %u tokens (sliced)\n", T_TOTAL);

    // 4. Caches + workspace.
    ie::KvCacheConfig kvcfg{};
    const uint32_t L_lin  = (cfg.n_layers / 4) * 3;
    const uint32_t L_full = cfg.n_layers - L_lin;
    kvcfg.n_layers_full = L_full;
    kvcfg.n_kv_heads    = cfg.n_kv_heads;
    kvcfg.max_ctx       = T_TOTAL + 64;
    kvcfg.head_dim      = cfg.head_dim;
    kvcfg.use_int8      = false;
    ie::KvCache kv;
    if (auto err = kv.init(alloc, kvcfg); !err.empty()) {
        std::fprintf(stderr, "kv: %s\n", err.c_str()); return 1;
    }
    ie::DeltaNetState dn;
    if (auto err = dn.init(alloc, ie::DeltaNetStateConfig{
            L_lin, cfg.ssm_n_v_heads, cfg.ssm_head_dim, cfg.ssm_head_dim,
            cfg.ssm_inner * 2, cfg.ssm_conv_kernel}); !err.empty()) {
        std::fprintf(stderr, "dn: %s\n", err.c_str()); return 1;
    }
    if (auto err = model.ensure_workspace(T_TOTAL); !err.empty()) {
        std::fprintf(stderr, "ws: %s\n", err.c_str()); return 1;
    }
    if (auto err = model.ensure_attn_partials(kvcfg.max_ctx); !err.empty()) {
        std::fprintf(stderr, "ap: %s\n", err.c_str()); return 1;
    }

    auto* d_logits = sycl::malloc_device<sycl::half>(cfg.vocab, q);
    auto* d_ids    = sycl::malloc_device<int32_t>(T_TOTAL, q);
    q.memcpy(d_ids, ids.data(), T_TOTAL * sizeof(int32_t)).wait();

    std::system(("mkdir -p " + dump_root).c_str());
    const std::string prefix_A  = dump_root + "/A";
    const std::string prefix_A2 = dump_root + "/A2";

#if !IE_VALIDATE_C_ONLY
    // ----- Path A: single forward(T=1024) -----
    std::printf("\nPath A: forward(T=%u, start_pos=0)\n", T_TOTAL);
    kv.reset(); dn.reset(q);
    model.set_dump_prefix(prefix_A);
    model.forward(q, d_ids, T_TOTAL, /*start_pos=*/0, kv, dn, d_logits).wait();
    std::printf("  done\n");

    // ----- Path B: 4 × forward(T=256) -----
    std::printf("\nPath B: %u × forward(T=%u)\n", N_CHUNKS, CHUNK);
    kv.reset(); dn.reset(q);
    for (uint32_t c = 0; c < N_CHUNKS; ++c) {
        const std::string prefix_B =
            dump_root + "/B_c" + std::to_string(c);
        model.set_dump_prefix(prefix_B);
        model.forward(q, d_ids + c * CHUNK, CHUNK,
                      /*start_pos=*/c * CHUNK, kv, dn, d_logits).wait();
        std::printf("  chunk %u done (start_pos=%u)\n", c, c * CHUNK);
    }

    // ----- Path A2: a second single forward(T=1024) — IDENTICAL inputs to path A.
    std::printf("\nPath A2: forward(T=%u, start_pos=0) — identical inputs to Path A\n",
                T_TOTAL);
    kv.reset(); dn.reset(q);
    model.set_dump_prefix(prefix_A2);
    model.forward(q, d_ids, T_TOTAL, /*start_pos=*/0, kv, dn, d_logits).wait();
    std::printf("  done\n");
#else
    std::printf("\n[IE_VALIDATE_C_ONLY=1] Skipping Paths A, B, A2 to avoid GPU-state perturbation.\n");
#endif

    // ----- Path C1: pure T=1 streaming forward × T_TOTAL iterations.
    // Step 11 (2026-05-03): does the engine produce deterministic output
    // when T=1 throughout?  Each forward(T=1) goes through the SPLIT-K
    // fa2_decode kernel, NOT the naive full_attention.  If C1 vs C2 is
    // BIT-EXACT, the bug is isolated to T>1 full_attention specifically
    // (workaround = pure-T=1 prefill, lossy on perf but produces correct
    // output).  If C1 vs C2 also diverges, the bug is also in fa2_decode
    // or stems from cache-state non-determinism, ruling out the
    // "T>1 full_attention only" hypothesis.
    //
    // Implementation note: dumps are gated by dump_prefix.  Setting it
    // on every iteration would write 1024 sets of dump files and only
    // the last iteration's data survives anyway.  We set it ONLY on
    // the final iteration (k=T_TOTAL-1, absolute position 1023) so the
    // resulting dumps capture the activations at the exact same absolute
    // position as Path A row 1023 and Path B chunk-3 row 255.  This makes
    // the comparison apples-to-apples.
    // Step 13 checkpoint iters — dense in (16, 64) where Step 12 showed
    // the first-divergence onset clusters, plus coarser later points
    // (1023 to keep the original anchor).  Captured AFTER forward(T=1, k).
    const std::vector<uint32_t> checkpoints = {
        0, 1, 4, 8, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64,
        80, 96, 128, 192, 256, 511, 1023
    };
    std::set<uint32_t> checkpoint_set(checkpoints.begin(), checkpoints.end());
    std::vector<StateHashes> c1_hashes, c2_hashes;
    c1_hashes.reserve(checkpoints.size());
    c2_hashes.reserve(checkpoints.size());
    std::vector<uint8_t> hash_host_buf;

    std::printf("\nPath C1: forward(T=1) × %u (start_pos = 0..%u, fresh state)\n",
                T_TOTAL, T_TOTAL - 1);
    kv.reset(); dn.reset(q);
    const std::string prefix_C1 = dump_root + "/C1";
    model.set_dump_prefix("");  // off for first 1023 iterations
    for (uint32_t k = 0; k < T_TOTAL; ++k) {
        if (k == T_TOTAL - 1) model.set_dump_prefix(prefix_C1);
        model.forward(q, d_ids + k, 1, /*start_pos=*/k, kv, dn, d_logits).wait();
        if (checkpoint_set.count(k)) {
            c1_hashes.push_back(capture_state(q, kv, dn, hash_host_buf));
        }
    }
    std::printf("  done\n");

    // ----- Path C2: identical to C1, second run for determinism test.
    std::printf("\nPath C2: forward(T=1) × %u (start_pos = 0..%u) — identical inputs to C1\n",
                T_TOTAL, T_TOTAL - 1);
    kv.reset(); dn.reset(q);
    const std::string prefix_C2 = dump_root + "/C2";
    model.set_dump_prefix("");
    for (uint32_t k = 0; k < T_TOTAL; ++k) {
        if (k == T_TOTAL - 1) model.set_dump_prefix(prefix_C2);
        model.forward(q, d_ids + k, 1, /*start_pos=*/k, kv, dn, d_logits).wait();
        if (checkpoint_set.count(k)) {
            c2_hashes.push_back(capture_state(q, kv, dn, hash_host_buf));
        }
    }
    std::printf("  done\n");

    model.set_dump_prefix("");  // turn off

    // first_div: A-vs-B per-layer first divergence index (only set in non-C-only mode).
    int first_div = -1;
#if !IE_VALIDATE_C_ONLY
    // ----- Compare per-layer at last position -----
    std::printf("\n=== Per-layer activation comparison ===\n");
    std::printf("  (A row %u  vs  B chunk %u row %u — both = absolute pos %u)\n",
                T_TOTAL - 1, N_CHUNKS - 1, CHUNK - 1, T_TOTAL - 1);
    std::printf("  L=embed L01..L40=post-layer residual L41=final-norm\n");
    std::printf("  H=%u\n\n", cfg.hidden);

    std::printf("  %-5s  %-12s  %-12s  %-12s  %-9s  %-9s  %-9s  %s\n",
                "layer", "max_abs", "max_rel", "L2_diff",
                "L2_A", "L2_B", "rel_L2", "status");
    std::printf("  %s\n",
                "-----  ------------  ------------  ------------  ---------  ---------  ---------  ------");

    for (int L = 0; L < N_DUMP_LAYERS; ++L) {
        char fa[64], fb[64];
        std::snprintf(fa, sizeof(fa), "%s_L%02d.bin", prefix_A.c_str(), L);
        std::snprintf(fb, sizeof(fb), "%s/B_c%u_L%02d.bin",
                      dump_root.c_str(), N_CHUNKS - 1, L);

        auto a = read_row(fa, T_TOTAL - 1, cfg.hidden);
        auto b = read_row(fb, CHUNK - 1, cfg.hidden);
        if (a.empty() || b.empty()) {
            std::printf("  L%02d   <missing>\n", L);
            continue;
        }
        auto cm = compare_rows(a, b);
        const double rel_l2 = (cm.a_l2 > 1e-12) ? cm.l2_diff / cm.a_l2 : 0.0;

        const char* status =
            (cm.max_abs_diff < 1e-3) ? "OK"   :
            (cm.max_abs_diff < 1e-1) ? "drift":
                                       "DIVERGE";
        if (cm.n_nan_a + cm.n_nan_b > 0) status = "NaN!";
        if (cm.n_inf_a + cm.n_inf_b > 0) status = "Inf!";

        std::printf("  L%02d    %-12.6g  %-12.6g  %-12.6g  %-9.3g  %-9.3g  %-9.3g  %s\n",
                    L, cm.max_abs_diff, cm.max_rel_diff, cm.l2_diff,
                    cm.a_l2, cm.b_l2, rel_l2, status);

        if (first_div < 0 && cm.max_abs_diff >= 1e-2) first_div = L;
    }

    if (first_div < 0) {
        std::printf("\n=== NO DIVERGENCE — chunking is bit-equivalent ===\n");
    } else {
        std::printf("\n=== FIRST DIVERGENCE: layer L%02d ===\n", first_div);
        std::printf("(first layer where max_abs_diff >= 1e-2 — bug lives in this layer's compute)\n");
    }

    // ----- Within-layer-3 intermediate bisect -----
    // qwen36.cpp dumps 8 intermediate buffers when L == 3, gated by
    // dump_prefix_.  Filenames: ${prefix}_<tag>03.bin, .meta carries "T W".
    std::printf("\n=== Within-layer-3 (first full-attn) intermediate bisect ===\n");
    std::printf("  (A row %u  vs  B chunk %u row %u — both = absolute pos %u)\n\n",
                T_TOTAL - 1, N_CHUNKS - 1, CHUNK - 1, T_TOTAL - 1);

    struct IntermediateInfo {
        const char* tag;       // matches dump_int tag in qwen36.cpp
        const char* descr;     // human-readable
    };
    const IntermediateInfo stages[] = {
        {"I1qproj_", "after gemv_q_T(attn_q)        — Q proj output (Q+gate fold-in)"},
        {"I2qsplt_", "after split_q_gate_per_head    — Q only (gate split off)"},
        {"I3qnorm_", "after rms_norm_f32w(attn_q_norm) — QK-Norm on Q"},
        {"I4knorm_", "after rms_norm_f32w(attn_k_norm) — QK-Norm on K"},
        {"I5qrope_", "after rope_partial(Q)          — Q + RoPE"},
        {"I6krope_", "after rope_partial(K)          — K + RoPE"},
        {"I7attno_", "after full_attention            — attention output (pre-gate)"},
        {"I8sigga_", "after sigmoid_gate              — attention output (post-gate)"},
    };

    auto read_meta_width = [](const std::string& path) -> uint32_t {
        std::ifstream m(path);
        if (!m) return 0;
        uint32_t T_meta = 0, W = 0;
        m >> T_meta >> W;
        return W;
    };

    std::printf("  %-9s  %-12s  %-12s  %-12s  %-9s  %-9s  %-9s  %-7s  %s\n",
                "stage", "max_abs", "max_rel", "L2_diff",
                "L2_A", "L2_B", "rel_L2", "status", "description");
    std::printf("  %s\n",
                "---------  ------------  ------------  ------------  ---------  ---------  ---------  -------  ----");

    int first_int_div = -1;
    int first_int_idx = -1;
    for (int s = 0; s < int(sizeof(stages) / sizeof(stages[0])); ++s) {
        char fa[256], fb[256], ma[256];
        std::snprintf(fa, sizeof(fa), "%s_%s03.bin", prefix_A.c_str(), stages[s].tag);
        std::snprintf(ma, sizeof(ma), "%s_%s03.meta", prefix_A.c_str(), stages[s].tag);
        std::snprintf(fb, sizeof(fb), "%s/B_c%u_%s03.bin",
                      dump_root.c_str(), N_CHUNKS - 1, stages[s].tag);
        const uint32_t W = read_meta_width(ma);
        if (W == 0) {
            std::printf("  %-9s  <missing or unreadable meta>\n", stages[s].tag);
            continue;
        }
        auto a = read_row(fa, T_TOTAL - 1, W);
        auto b = read_row(fb, CHUNK - 1, W);
        if (a.empty() || b.empty()) {
            std::printf("  %-9s  <missing dump>\n", stages[s].tag);
            continue;
        }
        auto cm = compare_rows(a, b);
        const double rel_l2 = (cm.a_l2 > 1e-12) ? cm.l2_diff / cm.a_l2 : 0.0;

        const char* status =
            (cm.max_abs_diff == 0.0)  ? "EXACT"  :
            (cm.max_abs_diff < 1e-4)  ? "ulp"    :
            (cm.max_abs_diff < 1e-3)  ? "OK"     :
            (cm.max_abs_diff < 1e-1)  ? "drift"  :
                                        "DIVERGE";
        if (cm.n_nan_a + cm.n_nan_b > 0) status = "NaN!";
        if (cm.n_inf_a + cm.n_inf_b > 0) status = "Inf!";

        std::printf("  %-9s  %-12.6g  %-12.6g  %-12.6g  %-9.3g  %-9.3g  %-9.3g  %-7s  %s\n",
                    stages[s].tag,
                    cm.max_abs_diff, cm.max_rel_diff, cm.l2_diff,
                    cm.a_l2, cm.b_l2, rel_l2, status, stages[s].descr);

        if (first_int_div < 0 && cm.max_abs_diff > 0.0) {
            first_int_div = s;
            first_int_idx = s;
        }
        (void)first_int_idx;
    }

    if (first_int_div < 0) {
        std::printf("\n=== NO intermediate divergence — bug must be in residual_add or "
                    "attn_norm input change ===\n");
    } else {
        std::printf("\n=== FIRST INTERMEDIATE DIVERGENCE: %s ===\n",
                    stages[first_int_div].tag);
        std::printf("    %s\n", stages[first_int_div].descr);
        std::printf("(first stage where max_abs > 0 — bug lives in the kernel(s) PRECEDING this dump)\n");
    }

    // ----- Cross-position spot check on L03 (residual after 3 DN layers) -----
    // To rule out chunk-dependent DN state at non-last positions: compare
    // single's L03 row P vs chunked's L03 row of the same ABSOLUTE pos P.
    // For P < 256:  single row P  vs  B_c0 row P.
    // For 256 <= P < 512: single row P  vs  B_c1 row (P - 256).
    // ... and so on.
    std::printf("\n=== L03 (post-DN-layer-3 residual) cross-position spot check ===\n");
    std::printf("  position    max_abs_diff   max_rel_diff   note\n");
    std::printf("  --------    -------------  -------------  ----\n");
    char fa_l03[256];
    std::snprintf(fa_l03, sizeof(fa_l03), "%s_L03.bin", prefix_A.c_str());
    const uint32_t spot_positions[] = {0, 100, 255, 256, 500, 511, 512, 767, 768, 1000, 1023};
    for (uint32_t p : spot_positions) {
        const uint32_t chunk_idx = p / CHUNK;
        const uint32_t row_in_chunk = p % CHUNK;
        char fb_l03[256];
        std::snprintf(fb_l03, sizeof(fb_l03), "%s/B_c%u_L03.bin",
                      dump_root.c_str(), chunk_idx);
        auto a = read_row(fa_l03, p, cfg.hidden);
        auto b = read_row(fb_l03, row_in_chunk, cfg.hidden);
        if (a.empty() || b.empty()) {
            std::printf("  pos=%-5u   <missing>\n", p);
            continue;
        }
        auto cm = compare_rows(a, b);
        std::printf("  pos=%-5u   %-13.6g  %-13.6g  (B chunk=%u row=%u)\n",
                    p, cm.max_abs_diff, cm.max_rel_diff, chunk_idx, row_in_chunk);
    }

    // ----- A1 vs A2 comparison (intrinsic kernel determinism test) -----
    std::printf("\n=== A1 vs A2 (same single forward(T=%u) twice) — kernel-determinism check ===\n",
                T_TOTAL);
    std::printf("  Both runs use IDENTICAL inputs (same tokens, same fresh state).\n");
    std::printf("  If A1 == A2 bit-exact: kernel is deterministic for fixed inputs.\n");
    std::printf("  If A1 != A2:           kernel itself has non-determinism.\n\n");

    std::printf("  %-9s  %-12s  %-12s  %-12s  %-7s  %s\n",
                "stage", "max_abs", "max_rel", "L2_diff", "status", "description");
    std::printf("  %s\n",
                "---------  ------------  ------------  ------------  -------  ----");

    // Compare residual dumps L00..L41 (full hidden state row 1023).
    int a_first_div = -1;
    for (int L = 0; L < N_DUMP_LAYERS; ++L) {
        char fa1[256], fa2[256];
        std::snprintf(fa1, sizeof(fa1), "%s_L%02d.bin", prefix_A.c_str(),  L);
        std::snprintf(fa2, sizeof(fa2), "%s_L%02d.bin", prefix_A2.c_str(), L);
        auto a1 = read_row(fa1, T_TOTAL - 1, cfg.hidden);
        auto a2 = read_row(fa2, T_TOTAL - 1, cfg.hidden);
        if (a1.empty() || a2.empty()) continue;
        auto cm = compare_rows(a1, a2);
        const char* status =
            (cm.max_abs_diff == 0.0) ? "EXACT" :
            (cm.max_abs_diff < 1e-4) ? "ulp"   :
            (cm.max_abs_diff < 1e-3) ? "OK"    :
            (cm.max_abs_diff < 1e-1) ? "drift" :
                                       "DIVERGE";
        // Only print non-EXACT rows + a few key ones to keep output compact.
        const bool key_layer = (L == 0 || L == 3 || L == 4 || L == 7 ||
                                L == 11 || L == 20 || L == 40 || L == 41);
        if (key_layer || cm.max_abs_diff > 0.0) {
            std::printf("  L%02d        %-12.6g  %-12.6g  %-12.6g  %-7s  %s\n",
                        L, cm.max_abs_diff, cm.max_rel_diff, cm.l2_diff, status,
                        L == 0  ? "post-embed" :
                        L <= 3  ? "DN layer" :
                        L == 4  ? "after first FULL-ATTN layer" :
                        L == 41 ? "post-final-norm (lm_head input)" :
                                  "post-layer residual");
        }
        if (a_first_div < 0 && cm.max_abs_diff > 0.0) a_first_div = L;
    }

    // Compare layer-3 intermediate dumps (which kernel makes A1 != A2)?
    std::printf("\n  --- within-layer-3 intermediate dumps ---\n");
    const struct { const char* tag; const char* descr; } a_stages[] = {
        {"I1qproj_", "after gemv_q_T(attn_q)"},
        {"I2qsplt_", "after split_q_gate_per_head"},
        {"I3qnorm_", "after rms_norm_f32w(attn_q_norm)"},
        {"I4knorm_", "after rms_norm_f32w(attn_k_norm)"},
        {"I5qrope_", "after rope_partial(Q)"},
        {"I6krope_", "after rope_partial(K)"},
        {"I7attno_", "after full_attention"},
        {"I8sigga_", "after sigmoid_gate"},
    };
    int a_int_first_div = -1;
    for (int s = 0; s < int(sizeof(a_stages) / sizeof(a_stages[0])); ++s) {
        char fa1[256], fa2[256], ma1[256];
        std::snprintf(fa1, sizeof(fa1), "%s_%s03.bin",  prefix_A.c_str(),  a_stages[s].tag);
        std::snprintf(fa2, sizeof(fa2), "%s_%s03.bin",  prefix_A2.c_str(), a_stages[s].tag);
        std::snprintf(ma1, sizeof(ma1), "%s_%s03.meta", prefix_A.c_str(),  a_stages[s].tag);
        std::ifstream m(ma1);
        uint32_t T_meta = 0, W = 0;
        if (m) m >> T_meta >> W;
        if (W == 0) continue;
        auto a1 = read_row(fa1, T_TOTAL - 1, W);
        auto a2 = read_row(fa2, T_TOTAL - 1, W);
        if (a1.empty() || a2.empty()) continue;
        auto cm = compare_rows(a1, a2);
        const char* status =
            (cm.max_abs_diff == 0.0) ? "EXACT" :
            (cm.max_abs_diff < 1e-4) ? "ulp"   :
            (cm.max_abs_diff < 1e-3) ? "OK"    :
            (cm.max_abs_diff < 1e-1) ? "drift" :
                                       "DIVERGE";
        std::printf("  %-9s  %-12.6g  %-12.6g  %-12.6g  %-7s  %s\n",
                    a_stages[s].tag,
                    cm.max_abs_diff, cm.max_rel_diff, cm.l2_diff, status,
                    a_stages[s].descr);
        if (a_int_first_div < 0 && cm.max_abs_diff > 0.0) a_int_first_div = s;
    }

    if (a_first_div < 0 && a_int_first_div < 0) {
        std::printf("\n=== A1 == A2 BIT-EXACT — kernel IS deterministic ===\n");
        std::printf("    Therefore the A vs B divergence above is purely chunk-size-dependent\n");
        std::printf("    (different launch shape produces different math).\n");
        std::printf("    Pivot: investigate what's different between T=256 and T=1024 launch\n");
        std::printf("           (probably IGC scheduling, FMA ordering, or per-shape codegen).\n");
    } else {
        std::printf("\n=== A1 != A2 — kernel itself is NON-DETERMINISTIC ===\n");
        if (a_int_first_div >= 0) {
            std::printf("    First non-deterministic stage: %s (%s)\n",
                        a_stages[a_int_first_div].tag,
                        a_stages[a_int_first_div].descr);
        }
        std::printf("    Pivot: investigate kernel-internal sources (cache read race,\n");
        std::printf("           native::exp, atomic ops, or scheduler-dependent compute).\n");
    }

    // ----- L04 cross-position A1 vs A2 (Step 9: confirm uniformity of
    //       intra-kernel non-determinism across positions, not just row 1023).
    // Steps 4-8 only verified A1 vs A2 at row 1023.  L04 = residual after
    // the first FULL-ATTN layer.  If A1 != A2 only at row 1023 but EXACT
    // elsewhere, the dump itself has a race.  If A1 != A2 at multiple
    // positions but with VARYING magnitude, the bug is per-position
    // (e.g. tile boundary).  If UNIFORM across all positions, the bug is
    // systemic (kernel-internal source we haven't isolated yet).
    std::printf("\n=== L04 (post-first-FULL-ATTN-layer residual) A1 vs A2 cross-position ===\n");
    std::printf("  (Both A1 row P and A2 row P are the SAME absolute position; full-attn\n");
    std::printf("   reads ALL ctx positions from K/V cache, so any non-determinism in\n");
    std::printf("   intermediate cache positions would manifest here at every dependent row.)\n\n");
    std::printf("  position    max_abs_diff   max_rel_diff   L2_diff       rel_L2     status\n");
    std::printf("  --------    -------------  -------------  ------------  ---------  ------\n");
    char fa1_l04[256], fa2_l04[256];
    std::snprintf(fa1_l04, sizeof(fa1_l04), "%s_L04.bin", prefix_A.c_str());
    std::snprintf(fa2_l04, sizeof(fa2_l04), "%s_L04.bin", prefix_A2.c_str());
    const uint32_t a_spots[] = {0, 100, 256, 500, 767, 1023};
    int n_pos_diverge = 0;
    int n_pos_exact = 0;
    double max_pos_max_abs = 0.0;
    double min_nonzero_pos_max_abs = 1e30;
    for (uint32_t p : a_spots) {
        auto a1 = read_row(fa1_l04, p, cfg.hidden);
        auto a2 = read_row(fa2_l04, p, cfg.hidden);
        if (a1.empty() || a2.empty()) {
            std::printf("  pos=%-5u   <missing>\n", p);
            continue;
        }
        auto cm = compare_rows(a1, a2);
        const double rel_l2 = (cm.a_l2 > 1e-12) ? cm.l2_diff / cm.a_l2 : 0.0;
        const char* status =
            (cm.max_abs_diff == 0.0) ? "EXACT" :
            (cm.max_abs_diff < 1e-4) ? "ulp"   :
            (cm.max_abs_diff < 1e-3) ? "OK"    :
            (cm.max_abs_diff < 1e-1) ? "drift" :
                                       "DIVERGE";
        std::printf("  pos=%-5u   %-13.6g  %-13.6g  %-12.6g  %-9.3g  %s\n",
                    p, cm.max_abs_diff, cm.max_rel_diff, cm.l2_diff, rel_l2, status);
        if (cm.max_abs_diff == 0.0) ++n_pos_exact;
        else {
            ++n_pos_diverge;
            if (cm.max_abs_diff > max_pos_max_abs) max_pos_max_abs = cm.max_abs_diff;
            if (cm.max_abs_diff < min_nonzero_pos_max_abs)
                min_nonzero_pos_max_abs = cm.max_abs_diff;
        }
    }

    std::printf("\n  Cross-position summary:\n");
    std::printf("    EXACT positions:     %d / %zu\n",
                n_pos_exact, sizeof(a_spots) / sizeof(a_spots[0]));
    std::printf("    DIVERGENT positions: %d / %zu\n",
                n_pos_diverge, sizeof(a_spots) / sizeof(a_spots[0]));
    if (n_pos_diverge > 0) {
        std::printf("    diverging-pos max_abs range: [%g, %g] (variance ratio %.1fx)\n",
                    min_nonzero_pos_max_abs, max_pos_max_abs,
                    max_pos_max_abs / min_nonzero_pos_max_abs);
    }
#endif  // !IE_VALIDATE_C_ONLY

    // ----- C1 vs C2 (pure T=1 streaming determinism test, Step 11) -----
    // For pure T=1 streaming, dump files contain a SINGLE row (T=1) at
    // the final iteration's absolute position (1023).  Read row 0.
    std::printf("\n=== C1 vs C2 (pure T=1 streaming × %u, two identical runs) — fa2_decode determinism ===\n",
                T_TOTAL);
    std::printf("  Each forward(T=1) goes through split-K fa2_decode (NOT naive full_attention).\n");
    std::printf("  C1 == C2 EXACT  → bug is isolated to T>1 full_attention.\n");
    std::printf("  C1 != C2        → bug is systemic (decode/cache also affected).\n\n");

    std::printf("  %-9s  %-12s  %-12s  %-12s  %-7s  %s\n",
                "stage", "max_abs", "max_rel", "L2_diff", "status", "description");
    std::printf("  %s\n",
                "---------  ------------  ------------  ------------  -------  ----");

    int c_first_div = -1;
    for (int L = 0; L < N_DUMP_LAYERS; ++L) {
        char fc1[256], fc2[256];
        std::snprintf(fc1, sizeof(fc1), "%s_L%02d.bin", prefix_C1.c_str(), L);
        std::snprintf(fc2, sizeof(fc2), "%s_L%02d.bin", prefix_C2.c_str(), L);
        // Pure T=1 streaming: each dump file is exactly 1 row tall.
        auto c1 = read_row(fc1, /*row=*/0, cfg.hidden);
        auto c2 = read_row(fc2, /*row=*/0, cfg.hidden);
        if (c1.empty() || c2.empty()) continue;
        auto cm = compare_rows(c1, c2);
        const char* status =
            (cm.max_abs_diff == 0.0) ? "EXACT" :
            (cm.max_abs_diff < 1e-4) ? "ulp"   :
            (cm.max_abs_diff < 1e-3) ? "OK"    :
            (cm.max_abs_diff < 1e-1) ? "drift" :
                                       "DIVERGE";
        const bool key_layer = (L == 0 || L == 3 || L == 4 || L == 7 ||
                                L == 11 || L == 20 || L == 40 || L == 41);
        if (key_layer || cm.max_abs_diff > 0.0) {
            std::printf("  L%02d        %-12.6g  %-12.6g  %-12.6g  %-7s  %s\n",
                        L, cm.max_abs_diff, cm.max_rel_diff, cm.l2_diff, status,
                        L == 0  ? "post-embed" :
                        L <= 3  ? "DN layer" :
                        L == 4  ? "after first FULL-ATTN layer (fa2_decode)" :
                        L == 41 ? "post-final-norm (lm_head input)" :
                                  "post-layer residual");
        }
        if (c_first_div < 0 && cm.max_abs_diff > 0.0) c_first_div = L;
    }

    // Within-layer-3 (first full-attn) intermediates for T=1 streaming.
    // For T=1 the "full_attention" path is replaced by fa2_decode in
    // qwen36.cpp's dispatch, so I7attno_ here is the fa2_decode output.
    std::printf("\n  --- within-layer-3 intermediate dumps (T=1 fa2_decode path) ---\n");
    const struct { const char* tag; const char* descr; } c_stages[] = {
        {"I1qproj_", "after gemv_q_T(attn_q)"},
        {"I2qsplt_", "after split_q_gate_per_head"},
        {"I3qnorm_", "after rms_norm_f32w(attn_q_norm)"},
        {"I4knorm_", "after rms_norm_f32w(attn_k_norm)"},
        {"I5qrope_", "after rope_partial(Q)"},
        {"I6krope_", "after rope_partial(K)"},
        {"I7attno_", "after fa2_decode (T=1) attention output"},
        {"I8sigga_", "after sigmoid_gate"},
    };
    int c_int_first_div = -1;
    for (int s = 0; s < int(sizeof(c_stages) / sizeof(c_stages[0])); ++s) {
        char fc1[256], fc2[256], mc1[256];
        std::snprintf(fc1, sizeof(fc1), "%s_%s03.bin",  prefix_C1.c_str(), c_stages[s].tag);
        std::snprintf(fc2, sizeof(fc2), "%s_%s03.bin",  prefix_C2.c_str(), c_stages[s].tag);
        std::snprintf(mc1, sizeof(mc1), "%s_%s03.meta", prefix_C1.c_str(), c_stages[s].tag);
        std::ifstream m(mc1);
        uint32_t T_meta = 0, W = 0;
        if (m) m >> T_meta >> W;
        if (W == 0) continue;
        // T=1 dump → row 0 is the only row.
        auto c1 = read_row(fc1, /*row=*/0, W);
        auto c2 = read_row(fc2, /*row=*/0, W);
        if (c1.empty() || c2.empty()) continue;
        auto cm = compare_rows(c1, c2);
        const char* status =
            (cm.max_abs_diff == 0.0) ? "EXACT" :
            (cm.max_abs_diff < 1e-4) ? "ulp"   :
            (cm.max_abs_diff < 1e-3) ? "OK"    :
            (cm.max_abs_diff < 1e-1) ? "drift" :
                                       "DIVERGE";
        std::printf("  %-9s  %-12.6g  %-12.6g  %-12.6g  %-7s  %s\n",
                    c_stages[s].tag,
                    cm.max_abs_diff, cm.max_rel_diff, cm.l2_diff, status,
                    c_stages[s].descr);
        if (c_int_first_div < 0 && cm.max_abs_diff > 0.0) c_int_first_div = s;
    }

    if (c_first_div < 0 && c_int_first_div < 0) {
        std::printf("\n=== C1 == C2 BIT-EXACT — pure T=1 streaming IS deterministic ===\n");
        std::printf("    Bug is isolated to T>1 full_attention specifically.\n");
        std::printf("    Workaround viable: pure-T=1 prefill (slow but correct).\n");
    } else {
        std::printf("\n=== C1 != C2 — pure T=1 streaming is ALSO non-deterministic ===\n");
        if (c_int_first_div >= 0) {
            std::printf("    First non-deterministic stage: %s (%s)\n",
                        c_stages[c_int_first_div].tag,
                        c_stages[c_int_first_div].descr);
        }
        std::printf("    Bug is systemic (HW-level / decode kernel / cache contamination).\n");
        std::printf("    No simple T-shape workaround.\n");
    }

    // ----- Step 12: per-iteration KV/DN hash comparison ------------------
    // Identify the FIRST iteration at which C1 and C2 diverge, and
    // (when divergent) which structure diverges first: KV-cache K, KV V,
    // DN recurrent state, or DN conv state.
    std::printf("\n=== Step 12: per-iteration KV/DN hash check (C1 vs C2, T=1 streaming) ===\n");
    std::printf("  Hashes the populated portion of K/V cache (per layer, per kv-head)\n");
    std::printf("  + entire DN state + DN conv state, captured AFTER forward(T=1, k)\n");
    std::printf("  for k in {%s}.\n",
                "0,1,4,16,64,256,511,1023");
    std::printf("  Equal hashes → bit-identical state.  First-divergent iter localises\n");
    std::printf("  the kernel call that introduces non-determinism.\n\n");

    std::printf("  %-5s  %-9s  %-18s  %-18s  %-7s  %s\n",
                "iter", "kv_len",
                "c1_combined", "c2_combined", "match", "first_diff");
    std::printf("  %s\n",
                "-----  ---------  ------------------  ------------------  -------  ----------");
    int first_diverge_idx = -1;
    for (size_t i = 0; i < checkpoints.size(); ++i) {
        if (i >= c1_hashes.size() || i >= c2_hashes.size()) break;
        const auto& a = c1_hashes[i];
        const auto& b = c2_hashes[i];
        const uint64_t ah = a.combined();
        const uint64_t bh = b.combined();
        // Earliest-component localisation.  DN comes BEFORE the corresponding
        // full-attn layer in the residual stream, so check DN first.
        std::string first_diff = "—";
        bool any_diff = false;
        for (uint32_t L = 0; L < a.n_layers_dn; ++L) {
            if (a.dn_state_per_layer[L] != b.dn_state_per_layer[L]) {
                first_diff = "DN_state[L=" + std::to_string(L) + "]";
                any_diff = true; break;
            }
            if (a.dn_conv_per_layer[L] != b.dn_conv_per_layer[L]) {
                first_diff = "DN_conv[L=" + std::to_string(L) + "]";
                any_diff = true; break;
            }
        }
        if (!any_diff) {
            for (uint32_t L = 0; L < a.n_layers_full; ++L) {
                if (a.k_per_layer[L] != b.k_per_layer[L]) {
                    first_diff = "K[kv=" + std::to_string(L) + "]";
                    any_diff = true; break;
                }
                if (a.v_per_layer[L] != b.v_per_layer[L]) {
                    first_diff = "V[kv=" + std::to_string(L) + "]";
                    any_diff = true; break;
                }
            }
        }
        // Combined-hash sanity (should match any_diff).
        (void)ah; (void)bh;
        const char* match = any_diff ? "DIFF" : "EXACT";
        const uint32_t kv_len = a.lengths[0];
        std::printf("  %-5u  %-9u  %016lx  %016lx  %-7s  %s\n",
                    checkpoints[i], kv_len,
                    (unsigned long)ah, (unsigned long)bh, match, first_diff.c_str());
        if (first_diverge_idx < 0 && any_diff) first_diverge_idx = int(i);
    }

    if (first_diverge_idx < 0) {
        std::printf("\n=== KV/DN state IDENTICAL at all checkpoints — divergence is in\n");
        std::printf("    the LAST iter's read path (workspace / scratch / register state),\n");
        std::printf("    not in the cache contents themselves. ===\n");
    } else {
        const uint32_t k = checkpoints[first_diverge_idx];
        std::printf("\n=== FIRST DIVERGENT CHECKPOINT: iter %u (kv_len=%u) ===\n",
                    k, c1_hashes[first_diverge_idx].lengths[0]);
        std::printf("    Means a kernel between iter %s and iter %u introduced\n",
                    first_diverge_idx == 0 ? "0" :
                        std::to_string(checkpoints[first_diverge_idx-1]).c_str(),
                    k);
        std::printf("    non-deterministic state.  Bisect the gap by tightening checkpoints.\n");
    }

    // Per-DN-layer + per-kv-layer detail at the first divergent checkpoint.
    if (first_diverge_idx >= 0) {
        const auto& a = c1_hashes[first_diverge_idx];
        const auto& b = c2_hashes[first_diverge_idx];
        std::printf("\n  Per-DN-layer detail @ iter %u  (model layer = i + (i/3)):\n",
                    checkpoints[first_diverge_idx]);
        std::printf("    %-5s  %-12s  %-12s\n", "dn_i", "state_match", "conv_match");
        std::printf("    -----  ------------  ------------\n");
        int first_dn_diff = -1;
        for (uint32_t L = 0; L < a.n_layers_dn; ++L) {
            const bool s_eq = (a.dn_state_per_layer[L] == b.dn_state_per_layer[L]);
            const bool c_eq = (a.dn_conv_per_layer [L] == b.dn_conv_per_layer [L]);
            if (!s_eq || !c_eq) {
                std::printf("    %-5u  %-12s  %-12s\n",
                            L, s_eq ? "EXACT" : "DIFF", c_eq ? "EXACT" : "DIFF");
            }
            if (first_dn_diff < 0 && (!s_eq || !c_eq)) first_dn_diff = int(L);
        }
        if (first_dn_diff < 0) {
            std::printf("    (all DN layers EXACT — divergence is purely in KV cache)\n");
        } else {
            std::printf("    First DN-layer divergence: dn_i=%d\n", first_dn_diff);
        }

        std::printf("\n  Per-kv-layer detail @ iter %u:\n",
                    checkpoints[first_diverge_idx]);
        std::printf("    %-5s  %-7s  %-7s  %-7s\n",
                    "kv_i", "K_match", "V_match", "model");
        std::printf("    -----  -------  -------  -------\n");
        int first_kv_diff = -1;
        for (uint32_t L = 0; L < a.n_layers_full; ++L) {
            const bool k_eq = (a.k_per_layer[L] == b.k_per_layer[L]);
            const bool v_eq = (a.v_per_layer[L] == b.v_per_layer[L]);
            if (!k_eq || !v_eq) {
                // model_layer = 4*kv_i + 3 (3DN+1FULL pattern)
                std::printf("    %-5u  %-7s  %-7s  L%-2u\n",
                            L, k_eq ? "EXACT" : "DIFF",
                            v_eq ? "EXACT" : "DIFF", 4 * L + 3);
            }
            if (first_kv_diff < 0 && (!k_eq || !v_eq)) first_kv_diff = int(L);
        }
        if (first_kv_diff < 0) {
            std::printf("    (all KV layers EXACT — divergence is purely in DN state)\n");
        } else {
            std::printf("    First KV-layer divergence: kv_i=%d (model layer %u)\n",
                        first_kv_diff, 4 * first_kv_diff + 3);
        }
    }

    // ----- Step 17: per-DN-layer recurrence-input comparison @ iter 1023.
    // qwen36.cpp dumps DN{q,k,v,g,b}_<dn_idx>.bin for every DN layer when
    // dump_prefix_ is set (which the C1/C2 paths enable only on the final
    // iter).  Compare C1 vs C2 input bytes per layer.
    //
    // Decisive question:
    //   For DN layers where SSM state diverges (cascade region), are the
    //   recurrence INPUTS also divergent or are they bit-identical?
    //   - Inputs EXACT, state DIFF → recurrence kernel is non-deterministic.
    //   - Inputs DIFF, state DIFF → some upstream kernel produced non-det
    //     output despite earlier audits saying otherwise.
    std::printf("\n=== Step 17: per-DN-layer recurrence-input comparison @ iter %u ===\n",
                T_TOTAL - 1);
    std::printf("  Inputs are q_in, k_in, v_in, g_in, beta_in (the 5 deltanet_recurrence args).\n");
    std::printf("  Files: ${prefix}_DN{q,k,v,g,b}_<dn_idx>.bin captured by qwen36.cpp.\n\n");
    std::printf("  %-5s  %-8s  %-8s  %-8s  %-8s  %-8s  %s\n",
                "dn_i", "q_match", "k_match", "v_match", "g_match", "b_match",
                "interpretation");
    std::printf("  %s\n",
                "-----  --------  --------  --------  --------  --------  --------------");
    const struct { const char* tag; uint32_t bytes; } dn_in_specs[] = {
        {"DNq", uint32_t(32u * 128u * sizeof(float))},   // SVH * SHD * fp32
        {"DNk", uint32_t(32u * 128u * sizeof(float))},
        {"DNv", uint32_t(32u * 128u * sizeof(float))},
        {"DNg", uint32_t(32u           * sizeof(float))},
        {"DNb", uint32_t(32u           * sizeof(float))},
    };
    int dn_n_full_exact = 0;
    int dn_n_state_div_input_exact = 0;  // smoking-gun count
    int dn_n_input_div = 0;
    for (uint32_t L = 0; L < 30; ++L) {
        bool match[5] = {true, true, true, true, true};
        bool all_present = true;
        for (int s = 0; s < 5; ++s) {
            char fc1[256], fc2[256];
            std::snprintf(fc1, sizeof(fc1), "%s_%s_%02u.bin",
                          prefix_C1.c_str(), dn_in_specs[s].tag, L);
            std::snprintf(fc2, sizeof(fc2), "%s_%s_%02u.bin",
                          prefix_C2.c_str(), dn_in_specs[s].tag, L);
            std::ifstream f1(fc1, std::ios::binary);
            std::ifstream f2(fc2, std::ios::binary);
            if (!f1 || !f2) { all_present = false; break; }
            std::vector<char> b1(dn_in_specs[s].bytes), b2(dn_in_specs[s].bytes);
            f1.read(b1.data(), dn_in_specs[s].bytes);
            f2.read(b2.data(), dn_in_specs[s].bytes);
            match[s] = (std::memcmp(b1.data(), b2.data(), dn_in_specs[s].bytes) == 0);
        }
        if (!all_present) {
            std::printf("  L%-4u  <missing dump files for this layer>\n", L);
            continue;
        }
        const bool any_input_diff = !(match[0] && match[1] && match[2] &&
                                       match[3] && match[4]);
        // Cross-reference with state hash at iter 1023.
        const auto& last_a = c1_hashes.back();
        const auto& last_b = c2_hashes.back();
        const bool state_diff =
            (L < last_a.n_layers_dn) &&
            (last_a.dn_state_per_layer[L] != last_b.dn_state_per_layer[L]);
        const char* interp = "—";
        if (!any_input_diff && !state_diff) {
            interp = "EXACT (no cascade here)";
            ++dn_n_full_exact;
        } else if (!any_input_diff && state_diff) {
            interp = "*** INPUTS EXACT, STATE DIFF — kernel non-deterministic ***";
            ++dn_n_state_div_input_exact;
        } else if (any_input_diff && state_diff) {
            interp = "INPUTS DIFF (cascade from earlier layer)";
            ++dn_n_input_div;
        } else {  // input diff but state exact — should be rare
            interp = "INPUTS DIFF, STATE EXACT (??)";
            ++dn_n_input_div;
        }
        std::printf("  L%-4u  %-8s  %-8s  %-8s  %-8s  %-8s  %s\n", L,
                    match[0] ? "EXACT" : "DIFF",
                    match[1] ? "EXACT" : "DIFF",
                    match[2] ? "EXACT" : "DIFF",
                    match[3] ? "EXACT" : "DIFF",
                    match[4] ? "EXACT" : "DIFF",
                    interp);
    }
    std::printf("\n  Summary @ iter %u:\n", T_TOTAL - 1);
    std::printf("    Full EXACT (inputs+state):                            %d\n", dn_n_full_exact);
    std::printf("    Inputs EXACT but state DIFF (kernel non-determinism): %d  ← smoking gun if > 0\n",
                dn_n_state_div_input_exact);
    std::printf("    Inputs DIFF (cascade or upstream non-det):            %d\n", dn_n_input_div);

#if 0  // Step 27/28 trace compare blocks removed in cleanup — see docs/bisect_step25_26_summary.md
    {
        constexpr uint32_t kChkpts[11] = {0,1,2,4,8,16,32,64,128,191,192};
        constexpr int      kNChkpts    = 11;
        constexpr int      kLaneLow    = IE_DN_DEBUG_LANE_LOW;
        constexpr int      kLaneHigh   = IE_DN_DEBUG_LANE_HIGH;
        constexpr int      kNLanes     = (kLaneHigh - kLaneLow);
        constexpr int      kNFields    = 14;
        constexpr size_t   kBytes      =
            size_t(kNChkpts) * size_t(kNLanes) * size_t(kNFields) * sizeof(float);
        const char* kFieldNames[kNFields] = {
            "q_self", "k_self", "v_self", "g_in_raw", "alpha", "beta_t",
            "S_lid_entry", "S_0_entry", "S_lid_after_decay", "kv_mem", "delta",
            "S_lid_after_update", "S_0_after_update", "out_v"
        };
        std::printf("\n=== Step 27: DN-debug-buffer compare (DN=%d, head=%d, lanes [%d,%d), %d chkpts) ===\n",
                    IE_DN_DEBUG_DN_IDX, IE_DN_DEBUG_HEAD,
                    kLaneLow, kLaneHigh, kNChkpts);
        const std::string c1_path = prefix_C1 + "_DN_DEBUG.bin";
        const std::string c2_path = prefix_C2 + "_DN_DEBUG.bin";
        std::ifstream f1(c1_path, std::ios::binary);
        std::ifstream f2(c2_path, std::ios::binary);
        if (!f1 || !f2) {
            std::printf("  <missing dump files — IE_DN_DEBUG_TRACE may be off>\n");
        } else {
            std::vector<char> b1(kBytes), b2(kBytes);
            f1.read(b1.data(), kBytes);
            f2.read(b2.data(), kBytes);
            const float* c1f = reinterpret_cast<const float*>(b1.data());
            const float* c2f = reinterpret_cast<const float*>(b2.data());
            bool found = false;
            for (int ci = 0; ci < kNChkpts && !found; ++ci) {
                for (int li = 0; li < kNLanes && !found; ++li) {
                    for (int fi = 0; fi < kNFields; ++fi) {
                        const size_t idx =
                            size_t(ci) * size_t(kNLanes) * size_t(kNFields) +
                            size_t(li) * size_t(kNFields) +
                            size_t(fi);
                        if (std::memcmp(&c1f[idx], &c2f[idx], sizeof(float)) != 0) {
                            std::printf("  FIRST DIFF:\n");
                            std::printf("    checkpoint slot   : %d (outer iter = %u)\n",
                                        ci, kChkpts[ci]);
                            std::printf("    DN layer (dn_idx) : %d\n", IE_DN_DEBUG_DN_IDX);
                            std::printf("    head (hh)         : %d\n", IE_DN_DEBUG_HEAD);
                            std::printf("    lane (vv)         : %d  (lane_local=%d)\n",
                                        kLaneLow + li, li);
                            std::printf("    field             : %s (idx=%d)\n",
                                        kFieldNames[fi], fi);
                            std::printf("    C1 value          : %.9g  (hex %a)\n",
                                        double(c1f[idx]), double(c1f[idx]));
                            std::printf("    C2 value          : %.9g  (hex %a)\n",
                                        double(c2f[idx]), double(c2f[idx]));
                            found = true;
                            break;
                        }
                    }
                }
            }
            if (!found) {
                std::printf("  ALL %zu BYTES IDENTICAL.\n", kBytes);
                std::printf("  Note: this proves equality ONLY for DN=%d / head=%d /\n",
                            IE_DN_DEBUG_DN_IDX, IE_DN_DEBUG_HEAD);
                std::printf("        lanes [%d,%d) at outer iters {0,1,2,4,8,16,32,64,128,191,192}.\n",
                            kLaneLow, kLaneHigh);
                std::printf("        It does NOT exonerate the recurrence kernel for other\n");
                std::printf("        layers/heads/lanes/iters. Re-run with widened gates to\n");
                std::printf("        cover the actual divergent (dn_idx, head, lane, iter).\n");
            }
        }
    }

    // ----- Step 28 (wide): upstream-producer trace bytewise compare.
    // Buffer layout: [iter_slot][dn_idx][producer_slot][32 bytes].
    // Walk in chronological order: iter → dn_idx → producer → element.
    // Slots 0..3 = fp16 (16 halfs); slots 4..6 = fp32 (8 floats).
    {
        constexpr uint32_t kIters[16] = {
            0, 1, 2, 4, 8, 16, 32, 64, 80, 128, 192, 256, 511, 512, 768, 1023
        };
        constexpr int    kNIters    = 16;
        constexpr int    kNDN       = 30;
        constexpr int    kNSlots    = 7;
        constexpr size_t kSlotBytes = 32;
        constexpr size_t kBytes =
            size_t(kNIters) * size_t(kNDN) * size_t(kNSlots) * kSlotBytes;
        struct SlotInfo { const char* name; bool is_fp16; };
        const SlotInfo slots[kNSlots] = {
            {"ws_x_normed_  (fp16, before attn_qkv gemv)",          true },
            {"conv_state_   (fp16, before depthwise_conv1d)",       true },
            {"ws_qkv_       (fp16, after  gemv_q_T(attn_qkv))",     true },
            {"ws_qkv_silu_  (fp16, after  depthwise_conv1d_causal)",true },
            {"ws_q_fp32_pre (fp32, after  cast_qkv_split)",         false},
            {"ws_q_fp32_    (fp32, after  repeat_interleave)",      false},
            {"ws_q_fp32_    (fp32, after  l2_norm_scale)",          false},
        };

        std::printf("\n=== Step 28 (wide): DN-upstream-producer trace compare ===\n");
        std::printf("  Iters: {0,1,2,4,8,16,32,64,80,128,192,256,511,512,768,1023}\n");
        std::printf("  DN layers: 0..29   producer slots: 0..6   slot size: 32 B\n");
        std::printf("  Buffer: %zu bytes per path.\n", kBytes);
        const std::string c1_path = prefix_C1 + "_DN_UPSTREAM.bin";
        const std::string c2_path = prefix_C2 + "_DN_UPSTREAM.bin";
        std::ifstream f1(c1_path, std::ios::binary);
        std::ifstream f2(c2_path, std::ios::binary);
        if (!f1 || !f2) {
            std::printf("  <missing dump files — IE_DN_UPSTREAM_TRACE may be off>\n");
        } else {
            std::vector<unsigned char> b1(kBytes), b2(kBytes);
            f1.read(reinterpret_cast<char*>(b1.data()), kBytes);
            f2.read(reinterpret_cast<char*>(b2.data()), kBytes);

            bool any_diff = false;
            for (int it_i = 0; it_i < kNIters && !any_diff; ++it_i) {
                for (int dn_i = 0; dn_i < kNDN && !any_diff; ++dn_i) {
                    for (int s = 0; s < kNSlots && !any_diff; ++s) {
                        const size_t slot_off =
                            ((size_t(it_i) * kNDN + size_t(dn_i)) *
                             kNSlots + size_t(s)) * kSlotBytes;
                        const unsigned char* p1 = b1.data() + slot_off;
                        const unsigned char* p2 = b2.data() + slot_off;
                        if (std::memcmp(p1, p2, kSlotBytes) == 0) continue;
                        any_diff = true;
                        std::printf("  FIRST DIFF:\n");
                        std::printf("    iter            : %u (slot %d)\n",
                                    kIters[it_i], it_i);
                        std::printf("    dn_idx          : %d\n", dn_i);
                        std::printf("    producer slot   : %d  %s\n",
                                    s, slots[s].name);
                        if (slots[s].is_fp16) {
                            const auto* h1 = reinterpret_cast<const uint16_t*>(p1);
                            const auto* h2 = reinterpret_cast<const uint16_t*>(p2);
                            for (int e = 0; e < 16; ++e) {
                                if (h1[e] != h2[e]) {
                                    sycl::half hC1, hC2;
                                    std::memcpy(&hC1, &h1[e], sizeof(uint16_t));
                                    std::memcpy(&hC2, &h2[e], sizeof(uint16_t));
                                    std::printf("    element idx     : %d  (byte offset %zu)\n",
                                                e, size_t(e) * sizeof(uint16_t));
                                    std::printf("    C1 raw u16      : 0x%04x  decoded fp16 : %.9g\n",
                                                unsigned(h1[e]), double(float(hC1)));
                                    std::printf("    C2 raw u16      : 0x%04x  decoded fp16 : %.9g\n",
                                                unsigned(h2[e]), double(float(hC2)));
                                    break;
                                }
                            }
                        } else {
                            const auto* fp1 = reinterpret_cast<const float*>(p1);
                            const auto* fp2 = reinterpret_cast<const float*>(p2);
                            for (int e = 0; e < 8; ++e) {
                                uint32_t u1, u2;
                                std::memcpy(&u1, &fp1[e], sizeof(uint32_t));
                                std::memcpy(&u2, &fp2[e], sizeof(uint32_t));
                                if (u1 != u2) {
                                    std::printf("    element idx     : %d  (byte offset %zu)\n",
                                                e, size_t(e) * sizeof(float));
                                    std::printf("    C1 fp32         : %.9g  (hex %a)  raw u32 0x%08x\n",
                                                double(fp1[e]), double(fp1[e]), u1);
                                    std::printf("    C2 fp32         : %.9g  (hex %a)  raw u32 0x%08x\n",
                                                double(fp2[e]), double(fp2[e]), u2);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            if (!any_diff) {
                std::printf("  ALL %zu BYTES IDENTICAL across all 16 iters × 30 DN layers × 7 slots.\n",
                            kBytes);
                std::printf("  Recurrence inputs are bytewise reproducible across C1 and C2 at every\n");
                std::printf("  captured (iter, dn_idx) — yet the C1/C2 hash table still diverges.\n");
                std::printf("  → divergence enters either between captured iters, or downstream of\n");
                std::printf("    deltanet_recurrence (gated_rms_norm / ssm_out_gemv / MoE / residual).\n");
            }
        }
    }
#endif  // Step 27/28 compare blocks (removed in cleanup)

    sycl::free(d_logits, q);
    sycl::free(d_ids,    q);
    return first_div < 0 ? 0 : 1;
}
