// tools/mimo26_nan_probe.cpp -- ie-mimo26-nan-probe: where a non-finite value enters MiMo-V2.6's forward on a long prefill
// (docs/mimo26/P7_FIX64_FIX70.md section 6: ie-mimo26-cache-test --big met a NaN in layer 47's residual at position 128,783
// of concatenated source text). Everything is scanned by BITS (mimo26_scan_*): icpx's default fast fp model may fold
// float NaN / inf checks away.
//   default     every layer's residual stream exported for every row of every chunk (set_feature_layers over all layers)
//               and scanned, and each chunk's last-row logits (with their FNV, for bit-identity across builds). At the first
//               chunk with a non-finite residual: which layers and rows, the first bad row's magnitude through the layers,
//               then the state cut back to the chunk's start (one chunk is exactly what the SWA rings allow) and the chunk
//               run again with the op probe on the first bad layer watching that row -- which op goes non-finite first --
//               and that row's MoE input + routing (moe_L<L>_pos<P>.bin, for tools/mimo26/expert_recompute.py) and MoE
//               output (moeout_L<L>_pos<P>.f32, for tools/mimo26/moe_agree.py) written to --dump DIR.
//   --watch P   the same op probe and dumps at position P (layer --layer, default 47) whether or not anything goes
//               non-finite; the prefill then goes on.
//   --decode-at P  prefill [0, P), then P's token as a ONE-ROW step (the decode route: T <= 8, int-dot experts) under the
//               op probe, then the state cut back to P and P as row 0 of a 64-row chunk (the prefill route, XMX experts)
//               -- the same row down both expert routes (dumps suffixed _decode / _chunk64).
//   --dflash    the serving path instead: the drafter on the last card, its target layers exported, add_context after
//               every chunk as mimo26_run_ids does. At a refusal (the #64 guard), the old code's outcome is emulated: the
//               same rows added with the non-finite values zeroed, the refused positions' ring slots overwritten with NaN
//               (what the old fp16 conversion made of them), and one draft().
//   usage: ie-mimo26-nan-probe <model_dir> <text_file> [--ctx N] [--upto P] [--offset K] [--dump DIR] [--layer L]
//                              [--watch P | --decode-at P | --dflash] [--cpu-leg]
// --offset K drops the first K tokens: the same content K positions earlier (content- or position-dependent?).
// --upto P prefills at most P positions. The CPU expert leg is off (IE_DS41_CPU_MISS=0: reruns are bit-reproducible)
// unless --cpu-leg (the serving default, whose split follows the stream-slot history).
#include "ie/expert_stream.hpp"      // ds4_expert_priority_read_layers
#include "ie/mimo26.hpp"
#include "ie/mimo26_dflash.hpp"
#include "ie/mimo26_forward.hpp"
#include "ie/mimo26_host_rules.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

uint64_t fnv(const float* v, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { uint32_t u; std::memcpy(&u, v + i, 4); h = (h ^ u) * 1099511628211ull; }
    return h;
}
// the largest FINITE element's index by bits (a float's order as an integer key), -1 when nothing is finite
int64_t argmax_finite(const float* v, size_t n) {
    int64_t best = -1; uint32_t key = 0;
    for (size_t i = 0; i < n; ++i) {
        uint32_t u; std::memcpy(&u, v + i, 4);
        if ((u & 0x7FFFFFFFu) >= 0x7F800000u) continue;
        const uint32_t k = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
        if (best < 0 || k > key) { key = k; best = int64_t(i); }
    }
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: ie-mimo26-nan-probe <model_dir> <text_file> [--ctx N] [--upto P] [--offset K] [--dump DIR] [--layer L] [--watch P | --decode-at P | --dflash] [--cpu-leg]\n");
        return 2;
    }
    const std::string dir = argv[1], text_file = argv[2];
    uint32_t ctx = 135000, upto = 0, offset = 0; std::string dump; bool use_df = false, cpu_leg = false;
    int layer_opt = -1; int64_t watch = -1, decode_at = -1;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--ctx") ctx = uint32_t(std::atol(val().c_str()));
        else if (a == "--upto") upto = uint32_t(std::atol(val().c_str()));
        else if (a == "--offset") offset = uint32_t(std::atol(val().c_str()));
        else if (a == "--dump") dump = val();
        else if (a == "--layer") layer_opt = std::atoi(val().c_str());
        else if (a == "--watch") watch = std::atoll(val().c_str());
        else if (a == "--decode-at") decode_at = std::atoll(val().c_str());
        else if (a == "--dflash") use_df = true;
        else if (a == "--cpu-leg") cpu_leg = true;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (!cpu_leg) setenv("IE_DS41_CPU_MISS", "0", 1);
    ie::Mimo26Model m;
    if (auto e = m.load(dir); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    ie::Tokenizer tok;
    if (auto e = tok.load_from_hf_json(dir + "/tokenizer.json", dir + "/tokenizer_config.json"); !e.empty()) { std::fprintf(stderr, "tokenizer: %s\n", e.c_str()); return 1; }
    std::ifstream tf(text_file, std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(tf)), std::istreambuf_iterator<char>());
    const std::vector<int32_t> all = tok.encode(text, false);
    if (all.size() <= offset + 1) { std::fprintf(stderr, "%zu tokens of text, offset %u\n", all.size(), offset); return 2; }
    uint32_t N = uint32_t(std::min<size_t>(all.size() - offset, ctx - 1));
    if (upto) N = std::min(N, upto);
    if (decode_at >= 0 && uint64_t(decode_at) + 64 > N) { std::fprintf(stderr, "--decode-at %lld needs 64 positions after it within %u\n", (long long)decode_at, N); return 2; }
    const std::vector<int32_t> ids(all.begin() + offset, all.begin() + offset + N);
    const auto& cfg = m.config();
    const uint32_t H = cfg.dim, V = cfg.vocab_size, nl = cfg.n_layers, TK = cfg.n_activated_experts;
    std::printf("text: %zu tokens; prefilling [%u, %u) of it at positions [0, %u), ctx %u, CPU expert leg %s, %s\n",
                all.size(), offset, offset + N, N, ctx, cpu_leg ? "on" : "off", use_df ? "drafter path" : "every layer exported");

    std::vector<sycl::device> devs;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) devs.push_back(d);
    }
    if (devs.empty()) { std::fprintf(stderr, "no Arc GPU\n"); return 1; }
    std::vector<std::unique_ptr<sycl::queue>> queues; std::vector<sycl::queue*> qs;
    for (const auto& d : devs) { queues.push_back(std::make_unique<sycl::queue>(sycl::context(d), d, sycl::property_list{sycl::property::queue::in_order{}})); qs.push_back(queues.back().get()); }
    ie::Mimo26Options opt; opt.max_ctx = ctx; opt.max_tokens = std::min<uint32_t>(2048, ctx); opt.n_static = 0;   // the engine's configuration
    if (const std::string rp = dir + "/ie_ranking_mimo26_chat.txt"; std::ifstream(rp).good())
        if (auto e = ie::ds4_expert_priority_read_layers(rp, cfg.n_routed_experts, nl, opt.ranking); !e.empty()) { std::fprintf(stderr, "ranking: %s\n", e.c_str()); return 1; }
    ie::Mimo26DFlash df;
    if (use_df) {                                                    // before the forward, as mimo26_load
        if (auto e = df.load(dir); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
        if (auto e = df.init(*qs.back(), m.embed.w->data, V, df.config().window); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
    }
    ie::Mimo26Forward fwd;
    if (auto e = fwd.init(qs, m, opt); !e.empty()) { std::fprintf(stderr, "init: %s\n", e.c_str()); return 1; }
    const uint32_t C = opt.max_tokens;
    std::vector<uint32_t> layers;
    if (use_df) { df.set_head(fwd.head_weights()); layers = df.config().target_layers; }
    else for (uint32_t L = 0; L < nl; ++L) layers.push_back(L);
    if (auto e = fwd.set_feature_layers(layers, use_df ? df.config().window : C); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }

    // the op probe's report for layer L watching absolute position P (written to --dump with `tag`)
    auto report = [&](const std::string& what, int L, uint32_t P, const std::string& tag) {
        std::printf("op probe, layer %d, position %u, %s:\n", L, P, what.c_str());
        std::printf("  %-10s %14s %24s %14s | %16s %14s\n", "op", "non-finite", "first (row, col)", "max finite|x|", "watched row n-f", "row max|x|");
        for (const auto& p : fwd.probe_ops()) {
            char first[48] = "-";
            if (p.all.first_bad >= 0) std::snprintf(first, sizeof first, "(%lld, %lld)", (long long)(p.all.first_bad / p.cols), (long long)(p.all.first_bad % p.cols));
            std::printf("  %-10s %14llu %24s %14.1f | %16llu %14.1f\n", p.op.c_str(), (unsigned long long)p.all.non_finite, first, double(p.all.max_abs),
                        (unsigned long long)p.row.non_finite, double(p.row.max_abs));
        }
        const auto& moe = fwd.probe_moe_row();
        if (moe.empty()) return;
        std::printf("  the watched row's routing:");
        for (uint32_t k = 0; k < TK; ++k) {
            int32_t e; float w;
            std::memcpy(&e, moe.data() + size_t(H) * 4 + k * 4, 4); std::memcpy(&w, moe.data() + size_t(H) * 4 + size_t(TK) * 4 + k * 4, 4);
            std::printf(" %d(%.5f)", e, double(w));
        }
        std::printf("\n");
        if (dump.empty()) return;
        std::error_code ec; std::filesystem::create_directories(dump, ec);
        const std::string stem = "_L" + std::to_string(L) + "_pos" + std::to_string(P) + tag;
        const std::string fi = dump + "/moe" + stem + ".bin", fo = dump + "/moeout" + stem + ".f32";
        std::ofstream(fi, std::ios::binary).write(reinterpret_cast<const char*>(moe.data()), std::streamsize(moe.size()));
        const auto& out = fwd.probe_moe_out();
        if (!out.empty()) std::ofstream(fo, std::ios::binary).write(reinterpret_cast<const char*>(out.data()), std::streamsize(out.size() * 4));
        std::printf("  written %s%s\n", fi.c_str(), out.empty() ? "" : (" and " + fo).c_str());
    };

    std::vector<float> lg;
    const uint32_t N0 = decode_at >= 0 ? uint32_t(decode_at) : N;   // --decode-at: the prefill stops at P
    for (uint32_t off = 0; off < N0; off += C) {
        const uint32_t n = std::min(C, N0 - off);
        if (auto e = fwd.forward(ids.data() + off, n, off, lg, false); !e.empty()) { std::printf("forward at %u: %s\n", off, e.c_str()); return 1; }
        const ie::Mimo26Scan ls = ie::mimo26_scan_f32(lg.data(), lg.size());
        const int64_t am = argmax_finite(lg.data(), lg.size());
        const uint32_t rows = fwd.feat_rows();
        const float* feat = fwd.features();
        char logit_note[200];
        std::snprintf(logit_note, sizeof logit_note, "last-row logits: %llu of %u non-finite, finite argmax %lld, fnv %016llx%s", (unsigned long long)ls.non_finite, V,
                      (long long)am, (unsigned long long)fnv(lg.data(), lg.size()), ls.non_finite == V ? " (greedy std::max_element would pick 0)" : "");

        if (use_df) {   // ---- the serving path: the drafter's context after every chunk ----
            const std::string e = df.add_context(feat, rows, off + n - rows, rows);
            std::printf("[%6u, %6u): add_context %s; %s\n", off, off + n, e.empty() ? "ok" : "REFUSED", logit_note);
            if (e.empty()) continue;
            std::printf("  the guard (#64): %s\n", e.c_str());
            // the old code's outcome: the same rows added (the non-finite values zeroed, so the guard passes), then the
            // positions that carried a non-finite feature poisoned in the ring as the old fp16 conversion would have
            const uint32_t nf = uint32_t(layers.size());
            std::vector<float> clean(feat, feat + size_t(nf) * rows * H);
            std::vector<uint32_t> bad_pos;
            for (uint32_t r = 0; r < rows; ++r) {
                bool bad = false;
                for (uint32_t f = 0; f < nf; ++f) {
                    float* row = clean.data() + (size_t(f) * rows + r) * H;
                    if (ie::mimo26_first_non_f16(row, H) < H) { bad = true; for (uint32_t c = 0; c < H; ++c) if (ie::mimo26_first_non_f16(row + c, 1) == 0) row[c] = 0.f; }
                }
                if (bad) bad_pos.push_back(off + n - rows + r);
            }
            if (const std::string e2 = df.add_context(clean.data(), rows, off + n - rows, rows); !e2.empty()) { std::printf("  emulation: add_context: %s\n", e2.c_str()); return 1; }
            for (uint32_t P : bad_pos) {
                std::vector<std::pair<void*, uint64_t>> spans;
                df.state_spans(P, P + 1, spans);
                for (const auto& [ptr, bytes] : spans) df.queue()->fill(ptr, uint16_t(0x7E00), bytes / 2);   // fp16 NaN
            }
            df.queue()->wait();
            std::vector<int32_t> d;
            const std::string e3 = df.draft(am >= 0 ? int32_t(am) : 0, df.config().block - 1, d, 0.f);
            std::printf("  the OLD code (no guard): %zu position(s) of NaN in the drafter's ring (first %u); the next draft: %s\n",
                        bad_pos.size(), bad_pos.empty() ? 0u : bad_pos.front(), e3.empty() ? ("ok, " + std::to_string(d.size()) + " drafts").c_str() : e3.c_str());
            return 0;
        }

        // ---- every layer exported: which layers and rows are non-finite ----
        int first_layer = -1; int64_t first_row = -1;
        uint32_t worst_layer = 0; float worst = 0.f;
        std::string bad_layers;
        for (uint32_t f = 0; f < nl; ++f) {
            const float* blk = feat + size_t(f) * rows * H;
            const ie::Mimo26Scan s = ie::mimo26_scan_f32(blk, size_t(rows) * H);
            if (s.max_at >= 0 && s.max_abs > worst) { worst = s.max_abs; worst_layer = f; }
            if (!s.non_finite) continue;
            uint32_t bad_rows = 0; int64_t r0 = -1;
            for (uint32_t r = 0; r < rows; ++r)
                if (ie::mimo26_scan_f32(blk + size_t(r) * H, H).non_finite) { ++bad_rows; if (r0 < 0) r0 = r; }
            if (first_layer < 0) { first_layer = int(f); first_row = r0; }
            bad_layers += " L" + std::to_string(f) + ":" + std::to_string(bad_rows) + "@" + std::to_string(off + n - rows + uint32_t(r0));
        }
        std::printf("[%6u, %6u): residual max finite |x| %.1f (layer %u); %s%s%s\n", off, off + n, double(worst), worst_layer, logit_note,
                    first_layer < 0 ? "" : "; NON-FINITE rows (layer:count@first position):", bad_layers.c_str());
        const bool watched = watch >= int64_t(off) && watch < int64_t(off + n);
        if (first_layer < 0 && !watched) continue;

        const int L = first_layer >= 0 ? (layer_opt >= 0 ? layer_opt : first_layer) : (layer_opt >= 0 ? layer_opt : int(nl) - 1);
        const uint32_t r = first_layer >= 0 && !watched ? uint32_t(first_row) : uint32_t(watch - int64_t(off)), P = off + n - rows + r;
        if (first_layer >= 0) {
            std::printf("\nFIRST NON-FINITE: layer %d, position %u (row %u of the chunk [%u, %u)); token id %d, text position %u\n", first_layer, P, r, off, off + n, ids[P], offset + P);
            std::printf("that row through the layers (max finite |x| / non-finite count):");
            for (uint32_t f = 0; f < nl; ++f) {
                const ie::Mimo26Scan s = ie::mimo26_scan_f32(feat + (size_t(f) * rows + r) * H, H);
                std::printf("%s L%u %.1f/%llu", f % 8 ? "" : "\n ", f, double(s.max_abs), (unsigned long long)s.non_finite);
            }
            std::printf("\n");
        }
        const uint64_t h_first = fnv(lg.data(), lg.size());
        // the chunk again, from the same state: one chunk back is exactly what the SWA rings hold (window + chunk slots)
        fwd.rewind(off);
        fwd.set_probe(L, int(r));
        if (auto e = fwd.forward(ids.data() + off, n, off, lg, false); !e.empty()) { std::printf("rerun: %s\n", e.c_str()); return 1; }
        report(std::string("the chunk run again (") + (fnv(lg.data(), lg.size()) == h_first ? "last-row logits reproduced bit for bit" : "NOT reproduced -- read with care") + ")", L, P, "");
        fwd.set_probe(-1, -1);
        if (first_layer >= 0) { std::printf("the rerun's last-row logits: %llu non-finite\n", (unsigned long long)ie::mimo26_scan_f32(lg.data(), lg.size()).non_finite); return 0; }
    }

    if (decode_at >= 0) {   // ---- the same row down both expert routes ----
        const uint32_t P = uint32_t(decode_at);
        const int L = layer_opt >= 0 ? layer_opt : int(nl) - 1;
        std::printf("\nposition %u (token id %d, text position %u) after a prefill of [0, %u):\n", P, ids[P], offset + P, P);
        fwd.set_probe(L, 0);
        if (auto e = fwd.forward(ids.data() + P, 1, P, lg, false); !e.empty()) { std::printf("decode step: %s\n", e.c_str()); return 1; }
        report("a one-row step (the decode route: int-dot experts, fp16 gate / up / SwiGLU stores)", L, P, "_decode");
        std::printf("  the step's logits: %llu non-finite\n", (unsigned long long)ie::mimo26_scan_f32(lg.data(), lg.size()).non_finite);
        fwd.rewind(P);
        fwd.set_probe(L, 0);
        if (auto e = fwd.forward(ids.data() + P, 64, P, lg, false); !e.empty()) { std::printf("64-row chunk: %s\n", e.c_str()); return 1; }
        report("row 0 of a 64-row chunk (the prefill route: XMX experts, fp32 gate / up, fp16 SwiGLU store)", L, P, "_chunk64");
        fwd.set_probe(-1, -1);
        return 0;
    }
    std::printf("no non-finite value in %u positions\n", N0);
    return 0;
}
