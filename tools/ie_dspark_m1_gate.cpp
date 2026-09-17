// tools/ie_dspark_m1_gate.cpp — dspark M1 gate: engine DsparkDrafter forward
// vs the PrismML fork's CPU golden vectors on identical tap inputs.
//
// GATE (plan §3 stage 5 / doc 08 M1):
//   * hidden-state cosine >= 0.999 on EVERY draft position, AND
//   * block argmax (post-markov draft id) match >= 30/32 positions.
// Also reports logits cosine + base-logit argmax match as secondary metrics.
//
// This tool loads the drafter (~5 GB F16 working set) onto ONE B70 — the caller
// MUST confirm the B70s are clear first (fuser -v /dev/dri/renderD129
// /dev/dri/renderD130). No target model is ever loaded.
//
// usage: ie-dspark-m1-gate <drafter.gguf> <ref0.bin> [ref1.bin ...]
//        [--cos <0.999>] [--argmax-rate <0.9375>] [--ordinal N] [--device NAME]

#include "ie/allocator.hpp"
#include "ie/dspark_drafter.hpp"
#include "ie/gguf.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr int32_t kMagic   = 0x44535052; // 'DSPR'
constexpr int32_t kVersion = 1;

struct RefRound {
    int32_t n_ctx_rows = 0, n_embd_cap = 0, n_embd = 0, block_size = 0, vocab = 0,
            markov_rank = 0, anchor_token = 0;
    std::vector<int32_t> ctx_pos, draft_tokens, draft_pos, ref_draft_ids;
    std::vector<float>   ctx_feat, ref_hidden, ref_logits;
    std::string path;
};

bool read_ref(const std::string& path, RefRound& r, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open " + path; return false; }
    int32_t hdr[10];
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    if (!f) { err = path + ": truncated header"; return false; }
    if (hdr[0] != kMagic)   { err = path + ": bad magic";   return false; }
    if (hdr[1] != kVersion) { err = path + ": bad version"; return false; }
    r.n_ctx_rows = hdr[2]; r.n_embd_cap = hdr[3]; r.n_embd = hdr[4];
    r.block_size = hdr[5]; r.vocab = hdr[6]; r.markov_rank = hdr[7];
    r.anchor_token = hdr[8];
    r.path = path;

    auto ri = [&](std::vector<int32_t>& v, size_t n) {
        v.resize(n); f.read(reinterpret_cast<char*>(v.data()), n * sizeof(int32_t));
    };
    auto rf = [&](std::vector<float>& v, size_t n) {
        v.resize(n); f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
    };
    ri(r.ctx_pos, r.n_ctx_rows);
    rf(r.ctx_feat, (size_t)r.n_ctx_rows * r.n_embd_cap);
    ri(r.draft_tokens, r.block_size);
    ri(r.draft_pos, r.block_size);
    rf(r.ref_hidden, (size_t)r.block_size * r.n_embd);
    rf(r.ref_logits, (size_t)r.block_size * r.vocab);
    ri(r.ref_draft_ids, r.block_size);
    if (!f) { err = path + ": truncated payload"; return false; }
    return true;
}

double cosine(const float* a, const float* b, size_t n) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; ++i) {
        dot += double(a[i]) * double(b[i]);
        na  += double(a[i]) * double(a[i]);
        nb  += double(b[i]) * double(b[i]);
    }
    if (na == 0 || nb == 0) return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

int argmax(const float* a, size_t n) {
    int best = 0; float bv = a[0];
    for (size_t i = 1; i < n; ++i) if (a[i] > bv) { bv = a[i]; best = int(i); }
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <drafter.gguf> <ref0.bin> [ref1.bin ...] "
            "[--cos F] [--argmax-rate F] [--ordinal N] [--device NAME]\n", argv[0]);
        return 2;
    }
    std::string model_path = argv[1];
    std::vector<std::string> refs;
    double cos_thresh = 0.999;
    double argmax_rate_thresh = 30.0 / 32.0;
    uint32_t ordinal = 0;
    std::string device = "B70";
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--cos" && i + 1 < argc) cos_thresh = atof(argv[++i]);
        else if (a == "--argmax-rate" && i + 1 < argc) argmax_rate_thresh = atof(argv[++i]);
        else if (a == "--ordinal" && i + 1 < argc) ordinal = uint32_t(atoi(argv[++i]));
        else if (a == "--device" && i + 1 < argc) device = argv[++i];
        else refs.push_back(a);
    }
    if (refs.empty()) { std::fprintf(stderr, "no ref files given\n"); return 2; }

    // ---- load all ref rounds, find max ctx-row count for scratch sizing ----
    std::vector<RefRound> rounds(refs.size());
    uint32_t max_ctx_rows = 0;
    for (size_t i = 0; i < refs.size(); ++i) {
        std::string err;
        if (!read_ref(refs[i], rounds[i], err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        max_ctx_rows = std::max<uint32_t>(max_ctx_rows, uint32_t(rounds[i].n_ctx_rows));
    }
    std::printf("loaded %zu ref rounds; max_ctx_rows=%u\n", rounds.size(), max_ctx_rows);

    // ---- device + drafter (loads ONLY the drafter onto one B70) ----
    ie::DeviceAllocator alloc;
    if (auto e = alloc.init(device, ordinal); !e.empty()) {
        std::fprintf(stderr, "device init failed: %s\n", e.c_str()); return 1;
    }
    std::printf("device: %s\n", alloc.device().get_info<sycl::info::device::name>().c_str());

    ie::GgufReader g;
    if (auto e = g.open(model_path); !e.empty()) {
        std::fprintf(stderr, "gguf open failed: %s\n", e.c_str()); return 1;
    }
    ie::DsparkDrafter drafter;
    if (auto e = drafter.load(alloc, g, max_ctx_rows); !e.empty()) {
        std::fprintf(stderr, "drafter load failed: %s\n", e.c_str()); return 1;
    }
    std::printf("drafter loaded: H=%u n_layer=%u heads=%u/%u HD=%u F=%u vocab=%u "
                "n_capture=%u block_size=%u mask=%d markov_rank=%u has_markov=%d\n",
                drafter.H, drafter.n_layer, drafter.n_head, drafter.n_head_kv,
                drafter.HD, drafter.F, drafter.vocab, drafter.n_capture,
                drafter.block_size, drafter.mask_token_id, drafter.markov_rank,
                int(drafter.has_markov));

    // ---- run + diff ----
    double min_hidden_cos = 2.0, min_logits_cos = 2.0;
    int    total_pos = 0, hidden_cos_pass = 0, logits_cos_pass = 0;
    int    argmax_match = 0, base_argmax_match = 0;
    std::printf("\n round  pos | hidden_cos  logits_cos | eng_id  ref_id  ok | eng_base ref_base\n");
    std::printf(  "-------------+-----------------------+---------------------+------------------\n");

    for (size_t ri = 0; ri < rounds.size(); ++ri) {
        RefRound& R = rounds[ri];
        std::vector<float> eng_hidden, eng_logits;
        std::vector<int32_t> eng_ids;
        std::string e = drafter.draft_block(
            R.ctx_feat.data(), R.ctx_pos.data(), R.n_ctx_rows,
            R.draft_tokens.data(), R.draft_pos.data(),
            eng_hidden, eng_logits, eng_ids);
        if (!e.empty()) { std::fprintf(stderr, "draft_block failed (%s): %s\n", R.path.c_str(), e.c_str()); return 1; }

        for (int p = 0; p < R.block_size; ++p) {
            const double hc = cosine(eng_hidden.data() + (size_t)p * R.n_embd,
                                     R.ref_hidden.data() + (size_t)p * R.n_embd, R.n_embd);
            const double lc = cosine(eng_logits.data() + (size_t)p * R.vocab,
                                     R.ref_logits.data() + (size_t)p * R.vocab, R.vocab);
            const int eng_id = eng_ids[p];
            const int ref_id = R.ref_draft_ids[p];
            const int eng_base = argmax(eng_logits.data() + (size_t)p * R.vocab, R.vocab);
            const int ref_base = argmax(R.ref_logits.data() + (size_t)p * R.vocab, R.vocab);
            const bool ok = (eng_id == ref_id);

            min_hidden_cos = std::min(min_hidden_cos, hc);
            min_logits_cos = std::min(min_logits_cos, lc);
            if (hc >= cos_thresh) hidden_cos_pass++;
            if (lc >= cos_thresh) logits_cos_pass++;
            if (ok) argmax_match++;
            if (eng_base == ref_base) base_argmax_match++;
            total_pos++;

            std::printf(" %5zu %4d | %9.6f  %9.6f | %6d  %6d  %s | %8d %8d\n",
                        ri, p, hc, lc, eng_id, ref_id, ok ? "Y" : "n", eng_base, ref_base);
        }
    }

    const double argmax_rate = double(argmax_match) / total_pos;
    std::printf("\n===== M1 GATE SUMMARY =====\n");
    std::printf("positions            : %d\n", total_pos);
    std::printf("hidden cos  min=%.6f  >=%.4f: %d/%d\n", min_hidden_cos, cos_thresh, hidden_cos_pass, total_pos);
    std::printf("logits cos  min=%.6f  >=%.4f: %d/%d\n", min_logits_cos, cos_thresh, logits_cos_pass, total_pos);
    std::printf("block argmax match   : %d/%d (%.4f)  [threshold %.4f]\n",
                argmax_match, total_pos, argmax_rate, argmax_rate_thresh);
    std::printf("base-logit argmax    : %d/%d\n", base_argmax_match, total_pos);

    const bool cos_ok    = (hidden_cos_pass == total_pos);
    const bool argmax_ok = (argmax_rate >= argmax_rate_thresh);
    const bool pass = cos_ok && argmax_ok;
    std::printf("\nGATE: %s  (hidden_cos_all=%s, argmax_rate=%s)\n",
                pass ? "PASS" : "FAIL",
                cos_ok ? "PASS" : "FAIL", argmax_ok ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
