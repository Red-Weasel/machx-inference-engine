// tools/hy4flash_mini.cpp — ie-hy4flash-mini: the Hy4-flash load check and the mini-model gate
// (docs/hy4flash/00_PORT_AND_ABLITERATION_PLAN.md, phases 1-2).
//
//   ie-hy4flash-mini <ie_dir> --load-only
//       binds every tensor of the engine directory (scripts/hy4flash_oracle/pack.py) and prints what it found
//   ie-hy4flash-mini <ie_dir> --oracle <dir> [--cards 1|2] [--split L] [--chunk C] [--decode-from K]
//                    [--static N] [--pinned N] [--tol 0.02] [--exact-experts] [--force] [--rows L]
//       runs the oracle's token ids (scripts/hy4flash_oracle/mini_model.py: <dir>/ids.i32, logits.f32, streams.f32) --
//       positions [0, K) in chunks of C rows, the rest one token at a time -- and compares every row's logits and the
//       streams entering every layer. Exit 0 = PASS, 3 = FAIL.
#include "ie/hy4flash.hpp"
#include "ie/hy4flash_forward.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {
template <class T> bool read_bin(const std::string& path, std::vector<T>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const std::streamsize n = f.tellg(); f.seekg(0);
    out.resize(size_t(n) / sizeof(T));
    return bool(f.read(reinterpret_cast<char*>(out.data()), std::streamsize(out.size() * sizeof(T))));
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: ie-hy4flash-mini <ie_dir> --load-only | --oracle <dir> [--cards N] [--split L] [--chunk C] [--decode-from K] [--static N] [--pinned N] [--tol X]\n"); return 2; }
    const std::string dir = argv[1];
    std::string oracle, along, emulate, ranking_file; bool load_only = false, q8 = false, q8_dense = false; uint32_t q8_slots = 0; uint32_t qb_gu = 8, qb_dn = 8;
    uint32_t n_cards = 1, split = 0, chunk = 0, big_chunk = 0, n_static = 0, n_pinned = 0; int decode_from = -1; double tol = 0.02; bool exact = false; int rows_layer = -1; bool force = false; uint32_t first_layers = 0;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> const char* { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--load-only") load_only = true;
        else if (a == "--oracle") oracle = val();
        else if (a == "--cards") n_cards = uint32_t(std::atoi(val()));
        else if (a == "--split") split = uint32_t(std::atoi(val()));
        else if (a == "--chunk") chunk = uint32_t(std::atoi(val()));
        else if (a == "--big-chunk") big_chunk = uint32_t(std::atoi(val()));   // (more than --chunk rows left: chunks of up to this many, in lent VRAM)
        else if (a == "--decode-from") decode_from = std::atoi(val());
        else if (a == "--static") n_static = uint32_t(std::atoi(val()));
        else if (a == "--pinned") n_pinned = uint32_t(std::atoi(val()));
        else if (a == "--tol") tol = std::atof(val());
        else if (a == "--exact-experts") exact = true;
        else if (a == "--q8") q8 = true;
        else if (a == "--q8-dense") q8_dense = true;
        else if (a == "--qbits") { const std::string v = val(); const size_t cm = v.find(','); qb_gu = uint32_t(std::atoi(v.c_str())); qb_dn = cm == std::string::npos ? qb_gu : uint32_t(std::atoi(v.c_str() + cm + 1)); }
        else if (a == "--q8-slots") q8_slots = uint32_t(std::atoi(val()));
        else if (a == "--ranking") ranking_file = val();
        else if (a == "--emulate") { emulate = val(); exact = true; }
        else if (a == "--rows") rows_layer = std::atoi(val());
        else if (a == "--force") force = true;
        else if (a == "--along") along = val();
        else if (a == "--first-layers") first_layers = uint32_t(std::atoi(val()));
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }

    ie::Hy4FlashModel m;
    if (auto e = m.load(dir); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    const auto& c = m.config();
    uint32_t n_idx = 0, n_moe = 0;
    for (uint32_t L = 0; L < c.n_layers; ++L) { n_idx += c.indexed[L]; n_moe += c.moe[L]; }
    std::printf("hy4-flash: %u layers (%u indexed x %u heads, %u sliding x %u heads, window %u), hidden %u, %u streams, vocab %u\n",
                c.n_layers, n_idx, c.n_heads, c.n_layers - n_idx, c.n_heads_swa, c.window, c.dim, c.hc_mult, c.vocab_size);
    std::printf("  MoE: %u layers x %u experts, top %u, latent %u x inter %u, route scale %.3f, swiglu limit %.1f; dense FFN inter %u\n",
                n_moe, c.n_experts, c.top_k, c.latent_dim, c.moe_inter_dim, double(c.route_scale), double(c.swiglu_limit), c.inter_dim);
    std::printf("  indexer: %u-token groups, %u heads x %u, recall %u groups = %u keys; rope theta %.0f (indexed) / %.0f (sliding)\n",
                c.ix_ratio, c.ix_heads, c.ix_head_dim, c.ix_group_topk, c.ix_budget(), double(c.theta_indexed), double(c.theta_swa));
    std::printf("  n-gram layers:");
    for (const auto& g : m.engrams()) {
        uint64_t rows = 0; for (uint64_t s : g.head_sizes) rows += s;
        std::printf(" [before layer %u: %llu rows x %u, lookup %u, multipliers %lld %lld ..]", g.id - 1, static_cast<unsigned long long>(rows), c.engram_head_dim(), g.table_size,
                    static_cast<long long>(g.multipliers[0]), static_cast<long long>(g.multipliers[1]));
    }
    std::printf("\n  source: %s\n  dense matrices as fp16: %.2f GiB; expert pack: %s; default norm eps %g\n", m.source_dir().c_str(),
                double(m.dense_bytes_f16()) / double(1ull << 30), m.has_pack() ? "complete" : "MISSING or partial", double(c.torch_default_eps));
    if (m.overlay_tensors()) std::printf("  OVERLAY: %zu source tensors replaced by overlay.safetensors\n", m.overlay_tensors());
    std::printf("  text tensors no role claimed: %zu", m.unclaimed().size());
    for (size_t i = 0; i < std::min<size_t>(m.unclaimed().size(), 6); ++i) std::printf("%s%s", i ? ", " : " (", m.unclaimed()[i].c_str());
    std::printf("%s\n", m.unclaimed().empty() ? "" : " ..)");
    if (load_only) return m.unclaimed().empty() ? 0 : 3;
    if (oracle.empty()) { std::fprintf(stderr, "nothing to run: pass --oracle <dir> or --load-only\n"); return 2; }

    std::vector<int32_t> ids; std::vector<float> ref_logits, ref_streams;
    if (!read_bin(oracle + "/ids.i32", ids) || (!first_layers && !read_bin(oracle + "/logits.f32", ref_logits)) || !read_bin(oracle + "/streams.f32", ref_streams)) {
        std::fprintf(stderr, "cannot read the oracle files in %s\n", oracle.c_str()); return 1; }
    // --first-layers K (the real-weights oracle): K + 1 stream entries (entering layers 0 .. K-1, leaving layer K-1), no logits
    const uint32_t T = uint32_t(ids.size()), V = c.vocab_size, nL = first_layers ? first_layers + 1 : c.n_layers; const size_t HH = size_t(c.hc_mult) * c.dim;
    if (first_layers >= c.n_layers) { std::fprintf(stderr, "--first-layers must be under the layer count\n"); return 2; }
    if ((!first_layers && ref_logits.size() != size_t(T) * V) || ref_streams.size() != size_t(nL) * T * HH) { std::fprintf(stderr, "oracle sizes do not fit this model\n"); return 1; }

    std::vector<sycl::device> devs;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) devs.push_back(d);
    }
    if (devs.size() < n_cards || n_cards == 0) { std::fprintf(stderr, "%zu Arc GPUs found, %u asked\n", devs.size(), n_cards); return 1; }
    std::vector<std::unique_ptr<sycl::queue>> queues; std::vector<sycl::queue*> qs;
    for (uint32_t i = 0; i < n_cards; ++i) {
        queues.push_back(std::make_unique<sycl::queue>(sycl::context(devs[i]), devs[i], sycl::property_list{sycl::property::queue::in_order{}}));
        qs.push_back(queues.back().get());
    }

    if (chunk == 0) chunk = T;
    const uint32_t K = decode_from < 0 ? T : std::min<uint32_t>(T, uint32_t(decode_from));
    ie::Hy4fOptions o; o.max_ctx = std::max<uint32_t>(T, 64); o.max_tokens = std::max<uint32_t>(chunk, 8); o.big_tokens = big_chunk; o.split_layer = split; o.n_static = n_static; o.n_pinned = n_pinned; o.exact_experts = exact; o.emulate = emulate; o.q8_experts = q8; if (q8_slots) o.q8_slots = q8_slots; o.q8_dense = q8_dense; o.q_bits_gu = qb_gu; o.q_bits_dn = qb_dn;
    if (!ranking_file.empty()) {      // [MoE layers][experts] u32, most-used first -> the forward's per-layer ranking
        std::ifstream rf(ranking_file, std::ios::binary); uint32_t n_moe_r = 0; for (uint32_t L = 0; L < c.n_layers; ++L) n_moe_r += c.moe[L];
        std::vector<uint32_t> flat(size_t(n_moe_r) * c.n_experts);
        if (!rf || !rf.read(reinterpret_cast<char*>(flat.data()), std::streamsize(flat.size() * 4))) { std::fprintf(stderr, "--ranking: cannot read %u x %u u32 from %s\n", n_moe_r, c.n_experts, ranking_file.c_str()); return 2; }
        o.ranking.assign(c.n_layers, std::vector<uint32_t>(c.n_experts)); uint32_t ord = 0;
        for (uint32_t L = 0; L < c.n_layers; ++L) {
            if (c.moe[L]) { std::copy(flat.begin() + size_t(ord) * c.n_experts, flat.begin() + size_t(ord + 1) * c.n_experts, o.ranking[L].begin()); ++ord; }
            else for (uint32_t e = 0; e < c.n_experts; ++e) o.ranking[L][e] = e;
        }
    } o.stop_after = first_layers;
    ie::Hy4fForward fw;
    if (auto e = fw.init(qs, m, o); !e.empty()) { std::fprintf(stderr, "init: %s\n", e.c_str()); return 1; }
    if (q8_dense) std::printf("non-expert matrices as 8-bit: %u\n", fw.dense_q8_mats());
    if (q8) { std::printf("expert slots a layer:"); for (size_t i = 0; i < size_t(n_cards); ++i) std::printf(" card %zu %u", i, fw.q8_slots_of(i)); std::printf("\n"); }
    for (size_t i = 0; i < fw.n_cards(); ++i) std::printf("card %zu: %.2f GiB on the device\n", i, double(fw.vram_bytes(i)) / double(1ull << 30));

    // the reference's discrete choices (expert picks, recalled groups): compared always, used with --force
    std::vector<int32_t> ref_routes, ref_recall; std::vector<float> ref_margin;
    uint32_t n_moe_l = 0, n_idx_l = 0;
    for (uint32_t L = 0; L < (first_layers ? first_layers : c.n_layers); ++L) { n_moe_l += c.moe[L]; n_idx_l += c.indexed[L]; }
    // an image span in the oracle (mini_model.py --image): the rows' 3-D positions, which rows take their own embedding, those embeddings
    std::vector<int32_t> mm_pos3; std::vector<uint8_t> mm_is; std::vector<float> mm_emb;
    if (read_bin(oracle + "/pos3.i32", mm_pos3) && mm_pos3.size() == size_t(T) * 3 && read_bin(oracle + "/is_embed.u8", mm_is) && mm_is.size() == T && read_bin(oracle + "/embeds.f32", mm_emb))
        std::printf("oracle with an image span: %zu rows take their own embedding and 3-D positions\n", mm_emb.size() / c.dim);
    else { mm_pos3.clear(); mm_is.clear(); mm_emb.clear(); }
    const bool have_routes = read_bin(oracle + "/routes.i32", ref_routes) && ref_routes.size() == size_t(n_moe_l) * T * c.top_k;
    const bool have_recall = read_bin(oracle + "/recall.i32", ref_recall) && ref_recall.size() == size_t(n_idx_l) * T * c.ix_group_topk;
    read_bin(oracle + "/route_margin.f32", ref_margin);
    if (have_routes) fw.set_reference_routes(ref_routes.data(), T, force, n_moe_l);
    if (have_recall) fw.set_reference_recall(ref_recall.data(), T, force);
    if (force && (!have_routes || (!have_recall && T > c.ix_budget()))) { std::fprintf(stderr, "--force: the oracle has no routes.i32 / recall.i32 for this run\n"); return 1; }
    std::vector<float> logits(size_t(T) * V), streams(size_t(nL) * T * HH), out, dump;
    fw.set_stream_dump(&dump);
    uint32_t pos = 0, calls = 0;
    while (pos < T) {
        const uint32_t n = pos < K ? (big_chunk > chunk && K - pos > chunk ? std::min(big_chunk, K - pos) : std::min(chunk, K - pos)) : 1u;
        dump.clear();
        ie::Hy4fRowInputs rin; const bool mm = !mm_pos3.empty();
        if (mm) {
            size_t before = 0; for (uint32_t t = 0; t < pos; ++t) before += mm_is[t];
            rin.pos3 = mm_pos3.data() + size_t(pos) * 3; rin.is_embed = mm_is.data() + pos; rin.embeds = mm_emb.data() + before * c.dim;
        }
        if (auto e = fw.forward(ids.data() + pos, n, out, true, mm ? &rin : nullptr); !e.empty()) { std::fprintf(stderr, "forward at position %u (%u rows): %s\n", pos, n, e.c_str()); return 1; }
        if (!first_layers) std::memcpy(logits.data() + size_t(pos) * V, out.data(), size_t(n) * V * 4);
        for (uint32_t L = 0; L < nL; ++L) std::memcpy(streams.data() + (size_t(L) * T + pos) * HH, dump.data() + size_t(L) * n * HH, size_t(n) * HH * 4);
        pos += n; ++calls;
    }
    const auto& st = fw.stats();
    std::printf("ran %u tokens in %u calls (chunks of %u up to position %u, then single tokens): %.0f ms (attention %.0f, FFN %.0f, n-gram %.0f, of which indexer %.0f)\n",
                T, calls, chunk, K, st.ms_total, st.ms_attn, st.ms_ffn, st.ms_engram, st.ms_indexer);

    if (!along.empty()) {
        // (the overlay check) how much of every stream lies along a direction: |S . r| / |S|, the worst row and stream of
        // each layer's input. An abliteration overlay built for r must bring this to rounding level in every layer.
        std::vector<float> r;
        if (!read_bin(along, r) || r.size() != c.dim) { std::fprintf(stderr, "--along: %s is not %u floats\n", along.c_str(), c.dim); return 1; }
        double rn = 0; for (float v : r) rn += double(v) * v; rn = std::sqrt(rn);
        std::printf("share of the streams along the direction (worst row and stream | mean), per layer input:\n");
        double worst_all = 0;
        for (uint32_t L = 0; L < nL; ++L) {
            double worst = 0, sum = 0;
            for (uint32_t t = 0; t < T; ++t)
                for (uint32_t i = 0; i < c.hc_mult; ++i) {
                    const float* sv = streams.data() + ((size_t(L) * T + t) * c.hc_mult + i) * c.dim;
                    double dot = 0, n2 = 0;
                    for (uint32_t d = 0; d < c.dim; ++d) { dot += double(sv[d]) * r[d]; n2 += double(sv[d]) * sv[d]; }
                    const double sh = std::fabs(dot) / (rn * std::sqrt(std::max(n2, 1e-300)));
                    worst = std::max(worst, sh); sum += sh;
                }
            std::printf("  layer %2u: %.2e | %.2e\n", L, worst, sum / (double(T) * c.hc_mult));
            worst_all = std::max(worst_all, worst);
        }
        std::printf("ALONG: worst share %.3e\n", worst_all);
        return 0;
    }
    // ---- the comparison ----
    {
        const auto& rd = fw.ref_diffs();
        double mmax = 0;
        for (const auto& [L, p] : rd.routes) {
            uint32_t ord = 0; for (uint32_t l = 0; l < L; ++l) ord += c.moe[l];
            if (ref_margin.size() == size_t(n_moe_l) * T) mmax = std::max(mmax, double(ref_margin[size_t(ord) * T + p]));
        }
        std::printf("discrete choices vs the reference (%s): expert picks differ on %zu of %u (layer, row) pairs (largest reference margin among them %.5f); recalled groups differ on %zu of %u\n",
                    force ? "the reference's used" : "the engine's own used", rd.routes.size(), rd.route_rows, mmax, rd.recalls.size(), rd.recall_rows);
    }
    auto rel = [](const float* a, const float* b, size_t n, double* max_abs) {
        double d2 = 0, b2 = 0, mx = 0;
        for (size_t i = 0; i < n; ++i) { const double d = double(a[i]) - double(b[i]); d2 += d * d; b2 += double(b[i]) * b[i]; mx = std::max(mx, std::fabs(d)); }
        if (max_abs) *max_abs = mx;
        return std::sqrt(d2 / std::max(b2, 1e-300));
    };
    std::printf("streams entering each layer, relative error (all rows | worst row):\n");
    int first_bad = -1;
    for (uint32_t L = 0; L < nL; ++L) {
        const float* a = streams.data() + size_t(L) * T * HH; const float* b = ref_streams.data() + size_t(L) * T * HH;
        double worst = 0; uint32_t wt = 0;
        for (uint32_t t = 0; t < T; ++t) { const double r = rel(a + size_t(t) * HH, b + size_t(t) * HH, HH, nullptr); if (r > worst) { worst = r; wt = t; } }
        const double all = rel(a, b, size_t(T) * HH, nullptr);
        if (first_layers && L == first_layers) std::printf("  leaving layer %u: %.2e | %.2e at row %u\n", L - 1, all, worst, wt);
        else std::printf("  layer %2u (%s%s): %.2e | %.2e at row %u\n", L, c.indexed[L] ? "indexed" : "sliding", c.moe[L] ? ", MoE" : ", dense", all, worst, wt);
        if (first_bad < 0 && !(worst < tol)) first_bad = int(L);
        if (int(L) == rows_layer) {       // (diagnostic) every row's error at this layer's input, per stream
            for (uint32_t t = 0; t < T; ++t) {
                std::printf("    row %3u:", t);
                for (uint32_t i = 0; i < c.hc_mult; ++i) std::printf(" %.1e", rel(a + size_t(t) * HH + size_t(i) * c.dim, b + size_t(t) * HH + size_t(i) * c.dim, c.dim, nullptr));
                std::printf("%s", t % 2 ? "\n" : "  |");
            }
        }
    }
    if (first_layers) {
        std::printf("FIRST %u LAYERS on these weights: %s (tolerance %.3g)\n", first_layers, first_bad < 0 ? "PASS" : "FAIL", tol);
        if (first_bad >= 0) std::printf("first stream entry off by more than the tolerance: %d\n", first_bad);
        return first_bad < 0 ? 0 : 3;
    }
    uint32_t top1 = 0, near_tie = 0; double worst = 0, sum = 0, worst_abs = 0; uint32_t wrow = 0; bool finite = true; double kl_sum = 0, kl_max = 0, nll_e = 0, nll_r = 0;
    for (uint32_t t = 0; t < T; ++t) {
        const float* a = logits.data() + size_t(t) * V; const float* b = ref_logits.data() + size_t(t) * V;
        for (uint32_t v = 0; v < V; ++v) if (!(std::fabs(a[v]) < 1e30f)) finite = false;
        double ma = 0; const double r = rel(a, b, V, &ma);
        {   // KL(reference || engine) of the next-token distributions, and the reference's own NLL of its top token
            double ma_ = a[0], mb_ = b[0];
            for (uint32_t v = 1; v < V; ++v) { ma_ = std::max(ma_, double(a[v])); mb_ = std::max(mb_, double(b[v])); }
            double sa = 0, sb = 0;
            for (uint32_t v = 0; v < V; ++v) { sa += std::exp(double(a[v]) - ma_); sb += std::exp(double(b[v]) - mb_); }
            const double la = ma_ + std::log(sa), lb = mb_ + std::log(sb);
            double kl = 0;
            for (uint32_t v = 0; v < V; ++v) { const double lpb = double(b[v]) - lb; if (lpb > -40) kl += std::exp(lpb) * (lpb - (double(a[v]) - la)); }
            kl_sum += kl; kl_max = std::max(kl_max, kl);
            if (t + 1 < T) {             // the NLL of the token that actually follows, under each
                const int32_t nx = ids[t + 1];
                nll_e += -(double(a[nx]) - la); nll_r += -(double(b[nx]) - lb);
            }
        }
        sum += r; worst_abs = std::max(worst_abs, ma);
        if (r > worst) { worst = r; wrow = t; }
        // top-1: the same token, or a near-tie of the reference's own (its two logits closer than twice this row's largest difference)
        const size_t ia = size_t(std::max_element(a, a + V) - a), ib = size_t(std::max_element(b, b + V) - b);
        if (ia == ib) ++top1;
        else if (double(b[ib]) - double(b[ia]) <= 2.0 * ma) ++near_tie;
    }
    std::printf("logits: relative error mean %.2e, worst %.2e (row %u), largest absolute difference %.4f; top-1 agrees on %u of %u rows, %u more are near-ties of the reference%s\n",
                sum / T, worst, wrow, worst_abs, top1, T, near_tie, finite ? "" : "; NON-FINITE values");
    std::printf("next-token distributions: KL(reference || engine) mean %.5f, worst row %.4f; NLL of the text's own next tokens: engine %.4f, reference %.4f (difference %+.4f) over %u predictions\n",
                kl_sum / T, kl_max, nll_e / std::max(1u, T - 1), nll_r / std::max(1u, T - 1), (nll_e - nll_r) / std::max(1u, T - 1), T - 1);
    const bool pass = finite && worst < tol && top1 + near_tie == T && first_bad < 0;
    if (!pass && first_bad >= 0) std::printf("first layer whose input is off by more than %.3g: layer %d (the fault is in layer %d or the n-gram step before it)\n", tol, first_bad, first_bad - 1);
    std::printf("MINI GATE: %s (tolerance %.3g)\n", pass ? "PASS" : "FAIL", tol);
    return pass ? 0 : 3;
}
