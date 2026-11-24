// tools/ie_inkling2_run.cpp — ie-inkling2-run: Inkling 2 Mini (inkling2_loop) phase-1 forward pass on one card.
//
// Loads the model (Inkling2Model, src/model/inkling2.cpp), runs one sequence of token ids through the 48 layer calls,
// prints the last position's top-10 logits, and optionally dumps every call's output for scripts/inkling2/gate.sh.
//
//   ie-inkling2-run --model <dir> (--ids a,b,c | --prompt "<text>") [--card N] [--ref-bf16]
//                   [--routes routes.json] [--teacher <goldens dir>] [--dump-calls <dir>] [--repeat N]
//   ie-inkling2-run --model <dir> --prompt "<text>" --tokenize-only        (no GPU)
//
//   --routes      JSON [48][T][8]: impose these expert picks per call and token (the reference's, SPEC 16.3)
//   --teacher     a goldens folder (callNN_layerLL_o.npy, float32 [1, T, 2304]): each call's output is replaced by the
//                 golden one before the next call reads it (per-call error, not accumulated)
//   --ref-bf16    emulate the reference's bf16 rounding (SPEC 9; for the per-call comparison only)
//   --dump-calls  writes callNN_layerLL.f32 (raw float32 [T, 2304]) for the 48 calls, logits_last.f32 (raw float32
//                 [200058]) and routes.json ({"own": [48][T][8], "used": ..., "margin": [48][T]})
//   --repeat N    run the forward N times (state reset in between) and report the time of each; the first run
//                 includes the SYCL JIT compile of the kernels
//   --prompt      tokenised with the checkpoint's tokenizer.json (ie::Tokenizer::load_from_hf_json, no BOS added)
//   --ids-from    a goldens2 meta.json: its "prompt_ids" (or "ids")
//
// Phase 2 (KV-cached path):
//   --generate N  prefill, then N tokens one at a time (greedy; --temp/--top-k/--top-p/--seed sample with the engine's
//                 host sampler, ie::Ds41Generator::sample); prints ids, text, prefill and decode tok/s
//   --step-routes a goldens2 routes.json ({"step0": [48][T][8], "step1": [48][1][8], ...}): impose the reference's picks
//                 on the prefill (step0) and on each decode step
//   --force-ids   a,b,c: feed these ids as the generated tokens instead of the picked ones (per-step comparison
//                 after a divergence); "meta" = the goldens2 meta.json's "generated"
//   --dump-features DIR  with --generate: the draft head's target features (mtp_config.target_layer_ids) of every
//                 position, features.f32 [positions][taps][hidden] fp32, and tokens.json {"ids": prompt + fed, "taps"}
//   --draft-check DIR GOLD  the draft head alone on a --dump-features DIR: drafts at every anchor of the generated part
//                 compared with scripts/inkling2/dspark_harness.py --out GOLD (same drafts, same accepted counts)
//   --spec        with --generate (greedy): the draft head drafts 7 tokens, one 8-row forward verifies them; prints the
//                 accepted counts and the speed; the ids must equal a plain --generate run's; --spec-k K verifies
//                 only the first K drafts
//   --dump-steps  DIR: stepK.f32 (logits, raw float32) for the prefill (K = 0) and every step, steps.json (ids, own picks)
//   --teacher-steps DIR: a goldens2 folder with hidden_stepK.npy [48, T_K, 2304]: teacher-force every layer call of every
//                 step (use with --force-ids meta); the calls' own outputs go to DIR(dump-steps)/callsK.f32
//   --chunk C / --max-chunk M   prefill chunk / the largest chunk (sets scratch + ring slack; default 256)
//   --ctx N       reserve the global entries for N positions up front
//   --equiv H     2b check: last-position logits of (whole prompt at once) vs (H tokens, then one at a time) vs chunked
//                 prefill at --chunk and at --chunk2
//   --mem-table   after the load, the device bytes at 8K / 32K / 131K context and the largest context that fits
//   --truncate-check D  prefill the prompt, feed D more (junk) tokens, truncate back to (prompt - D) -- a rewind of 2D,
//                 so 2D <= rollback -- refeed the last D prompt tokens: the logits must equal the straight prompt run's; also checks the refusal below the
//                 earliest restorable position
//   --rollback R  how far truncate() may rewind (default = --max-chunk)
//   --bench a,b,..        decode tok/s at these context lengths (zero-filled context, 3 warm-up + 16 timed steps each)
//   --bench-prefill a,b,..  prefill tok/s of the first a, b, ... prompt ids (after a 300-token warm-up forward; the
//                 prompt must be long enough)
//   --bench-batch a,b,..  batched decode (Inkling2Model::batch_step) with a, b, ... conversations at once, their contexts
//                 spread evenly over --bench-ctx LO,HI (default 500,3000; zero-filled): total and per-conversation tok/s,
//                 3 warm-up + 16 timed steps each, and the device bytes
//   --image FILE  an image for the prompt's placeholder tokens (<|unused_200054|>, one per 40 x 40 block, in order; the
//                 count must match): encoded by the vision tower, its rows replace the placeholders
//   --dump-vision DIR  with --image: tower_embed / tower_layer0 / tower_last / image_features as raw fp32 files
//   --tp          two cards on one conversation (Inkling2LoadOptions::tp_queue): --card and the other card; every
//                 single-conversation mode works (not --parallel / --bench-batch / --profile)
//   --profile     with --generate: per-kernel GPU time over the decode steps (queue profiling) and GPU busy vs wall time
//
// Ablation sets (include/ie/inkling2.hpp Inkling2Ablation) and the abliteration batch mode:
//   --ablation P  an ablation json for every forward ("off" = none; default: <model>/ie_ablation.json if present)
//   --batch F     a .jsonl of {"name": .., "text": .. [, "append_ids": [..]]}: each text is ONE user turn rendered with the
//                 serving renderer, then the append_ids (e.g. the start of a recorded reply) after it
//                 (inkling2_render_prompt, --effort E: off | none | minimal | low | medium | high | xhigh | max |
//                 default = the template's 0.9; --open-turn appends <|message_model|>), state reset per prompt
//   --acts-out F  append, per prompt, the recorder's [--acts-last N (1)][48 calls][4 kinds][2304] float32 (kinds: partial
//                 state after the call, attention write, MoE write, MoE read-out; Inkling2RunOptions::record_last);
//                 --acts-mean writes the mean over those N positions instead ([48][4][2304] per prompt)
//   --gen N --gen-out F   greedy reply of up to N tokens per prompt (stops at the serving stop tokens); one json line per
//                 (candidate, prompt): ids, the parsed answer ("text"), reasoning, whether the turn ended, and -- for every
//                 candidate after the first -- the KL(first candidate || this) of the next-token distribution at the
//                 first --kl-steps S (16) positions of the FIRST candidate's reply, teacher-forced ("kl")
//   --ablate-list F  name<TAB>ablation json (or "none"), one candidate per line, all prompts per candidate
//   --kl-ref-out F / --kl-ref-in F   save the first candidate's replies + logits (the KL reference) / compare every
//                 candidate with a saved one (same batch file) instead of with the first candidate
//   --parallel B  decode up to B prompts at once on the card (Inkling2Model::batch_*): the same outputs as one at a
//                 time (each conversation's arithmetic is the same as alone, bit for bit), written in prompt order; a
//                 finished prompt's slot takes the next prompt
#include "ie/deepseek41_generate.hpp"
#include "ie/inkling2.hpp"
#include "ie/inkling2_vision.hpp"
#include "ie/inkling2_draft.hpp"
#include "ie/inkling2_engine.hpp"
#include "ie/kernel_profiler.hpp"
#include "ie/tokenizer.hpp"

#include "../third_party/nlohmann/json.hpp"

#include <cstdlib>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

// Minimal .npy reader: little-endian float32, C order. Returns "" on success.
std::string read_npy_f32(const std::string& path, std::vector<float>& out, std::vector<int64_t>& shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "cannot open " + path;
    char magic[6];
    f.read(magic, 6);
    if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0) return path + ": not a .npy file";
    uint8_t ver[2];
    f.read(reinterpret_cast<char*>(ver), 2);
    uint32_t hlen = 0;
    if (ver[0] == 1) { uint16_t h; f.read(reinterpret_cast<char*>(&h), 2); hlen = h; }
    else { f.read(reinterpret_cast<char*>(&hlen), 4); }
    std::string hdr(hlen, '\0');
    f.read(hdr.data(), hlen);
    if (hdr.find("'descr': '<f4'") == std::string::npos) return path + ": dtype is not <f4";
    if (hdr.find("'fortran_order': False") == std::string::npos) return path + ": fortran order not supported";
    const size_t a = hdr.find("'shape': ("), b = hdr.find(')', a);
    if (a == std::string::npos || b == std::string::npos) return path + ": no shape";
    shape.clear();
    std::stringstream ss(hdr.substr(a + 10, b - a - 10));
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        tok.erase(std::remove_if(tok.begin(), tok.end(), ::isspace), tok.end());
        if (!tok.empty()) shape.push_back(std::stoll(tok));
    }
    int64_t n = 1;
    for (auto d : shape) n *= d;
    out.resize(size_t(n));
    f.read(reinterpret_cast<char*>(out.data()), n * 4);
    if (!f) return path + ": short read";
    return {};
}

std::string golden_file(const std::string& dir, int i) {
    char pre[32];
    std::snprintf(pre, sizeof pre, "call%02d_layer", i);
    std::vector<std::string> hits;
    for (const auto& e : fs::directory_iterator(dir)) {
        const std::string n = e.path().filename().string();
        if (n.rfind(pre, 0) == 0 && n.size() > 4 && n.substr(n.size() - 4) == ".npy") hits.push_back(e.path().string());
    }
    std::sort(hits.begin(), hits.end());
    for (const auto& h : hits) if (h.size() > 6 && h.substr(h.size() - 6) == "_o.npy") return h;
    return hits.empty() ? std::string() : hits[0];
}

bool write_raw(const std::string& path, const float* p, size_t n) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(p), std::streamsize(n * 4));
    return bool(f);
}


std::vector<std::vector<std::array<int32_t, 8>>> parse_routes(const json& j, uint32_t NC, uint32_t T, std::string& err) {
    std::vector<std::vector<std::array<int32_t, 8>>> routes;
    if (!j.is_array() || j.size() != NC) { err = "routes: need [" + std::to_string(NC) + "][T][8]"; return {}; }
    for (const auto& cj : j) {
        if (!cj.is_array() || cj.size() != T) { err = "routes: need " + std::to_string(T) + " tokens per call"; return {}; }
        std::vector<std::array<int32_t, 8>> cr;
        for (const auto& tj : cj) {
            if (!tj.is_array() || tj.size() != 8) { err = "routes: need 8 experts per token"; return {}; }
            std::array<int32_t, 8> r{};
            for (int k = 0; k < 8; ++k) r[k] = tj[k].get<int32_t>();
            cr.push_back(r);
        }
        routes.push_back(std::move(cr));
    }
    return routes;
}

std::vector<int32_t> top_n(const std::vector<float>& lg, int n) {
    std::vector<int32_t> order(lg.size());
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + n, order.end(), [&](int32_t a, int32_t b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); });
    order.resize(n);
    return order;
}

void rel_err(const std::vector<float>& a, const std::vector<float>& b, double& rel, double& mx) {
    double n = 0, d = 0;
    mx = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double e = double(a[i]) - double(b[i]);
        d += e * e; n += double(b[i]) * double(b[i]); mx = std::max(mx, std::fabs(e));
    }
    rel = std::sqrt(d / n);
}

json load_json(const std::string& path) {
    std::ifstream f(path);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    json j;
    try { f >> j; } catch (const std::exception& e) { std::fprintf(stderr, "%s: %s\n", path.c_str(), e.what()); std::exit(1); }
    return j;
}

std::vector<int32_t> parse_ids(const std::string& s) {
    std::vector<int32_t> v;
    for (size_t p = 0; p < s.size();) { size_t c = s.find(',', p); if (c == std::string::npos) c = s.size(); v.push_back(std::atoi(s.substr(p, c - p).c_str())); p = c + 1; }
    return v;
}

}  // namespace

// The abliteration batch mode (see the header comment).
struct BatchPrompt { std::string name; std::vector<int32_t> ids; };

static double kl_logits(const std::vector<float>& p, const std::vector<float>& q) {     // KL(p || q) of softmaxes
    const float mp = *std::max_element(p.begin(), p.end()), mq = *std::max_element(q.begin(), q.end());
    double zp = 0, zq = 0;
    for (size_t i = 0; i < p.size(); ++i) { zp += std::exp(double(p[i]) - mp); zq += std::exp(double(q[i]) - mq); }
    const double lzp = std::log(zp) + mp, lzq = std::log(zq) + mq;
    double kl = 0;
    for (size_t i = 0; i < p.size(); ++i) {
        const double lp = double(p[i]) - lzp;
        kl += std::exp(lp) * (lp - (double(q[i]) - lzq));
    }
    return std::max(0.0, kl);
}

static int32_t argmax_row(const float* p, size_t n) {                   // top_n(.., 1): the lowest id among equal maxima
    size_t b = 0;
    for (size_t i = 1; i < n; ++i) if (p[i] > p[b]) b = i;
    return int32_t(b);
}

static int run_batch(ie::Inkling2Model& m, const ie::Tokenizer& tok, const ie::Inkling2Config& cfg, const std::vector<BatchPrompt>& P,
                     const std::string& ablate_list, const std::string& acts_out, uint32_t acts_last, uint32_t gen_n,
                     const std::string& gen_out, uint32_t kl_steps, bool acts_mean, const std::string& kl_ref_in,
                     const std::string& kl_ref_out, uint32_t parallel) {
    struct Cand { std::string name, path; };
    std::vector<Cand> cands;
    if (!ablate_list.empty()) {
        std::ifstream f(ablate_list);
        if (!f) { std::fprintf(stderr, "cannot open %s\n", ablate_list.c_str()); return 1; }
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            const size_t t = line.find('\t');
            if (t == std::string::npos) { std::fprintf(stderr, "%s: need name<TAB>path\n", ablate_list.c_str()); return 1; }
            cands.push_back({line.substr(0, t), line.substr(t + 1)});
        }
    } else cands.push_back({m.ablation_summary().empty() ? "none" : "file", ""});
    std::vector<int32_t> stop;
    for (const char* s : {"<|content_model_end_sampling|>", "<|endoftext|>", "<|message_user|>", "<|message_system|>", "<|message_tool|>"})
        if (const int32_t v = tok.find_token(s); v >= 0) stop.push_back(v);
    std::ofstream acts, gens;
    if (!acts_out.empty()) { acts.open(acts_out, std::ios::binary | std::ios::app); if (!acts) { std::fprintf(stderr, "cannot write %s\n", acts_out.c_str()); return 1; } }
    if (!gen_out.empty()) { gens.open(gen_out, std::ios::app); if (!gens) { std::fprintf(stderr, "cannot write %s\n", gen_out.c_str()); return 1; } }
    std::vector<std::vector<int32_t>> first_ids(P.size());                 // the first candidate's replies (KL reference)
    std::vector<std::vector<std::vector<float>>> first_lg(P.size());       // its logits at the first kl_steps positions
    // --kl-ref-in: the reference (ids + logits) saved by an earlier --kl-ref-out run over the SAME batch file; every
    // candidate of this run is then compared with it (none needs to run again)
    const bool ref_loaded = !kl_ref_in.empty();
    if (ref_loaded) {
        std::ifstream f(kl_ref_in, std::ios::binary);
        if (!f) { std::fprintf(stderr, "cannot open %s\n", kl_ref_in.c_str()); return 1; }
        uint32_t np = 0, vocab = 0;
        f.read(reinterpret_cast<char*>(&np), 4); f.read(reinterpret_cast<char*>(&vocab), 4);
        if (np != P.size()) { std::fprintf(stderr, "%s: %u prompts, the batch has %zu\n", kl_ref_in.c_str(), np, P.size()); return 1; }
        for (size_t pi = 0; pi < P.size(); ++pi) {
            uint32_t n = 0, st = 0;
            f.read(reinterpret_cast<char*>(&n), 4); first_ids[pi].resize(n); f.read(reinterpret_cast<char*>(first_ids[pi].data()), n * 4);
            f.read(reinterpret_cast<char*>(&st), 4); first_lg[pi].assign(st, std::vector<float>(vocab));
            for (auto& v : first_lg[pi]) f.read(reinterpret_cast<char*>(v.data()), std::streamsize(vocab) * 4);
        }
        if (!f) { std::fprintf(stderr, "%s: short read\n", kl_ref_in.c_str()); return 1; }
    }
    ie::Inkling2RunOptions ro;
    ro.record_routes = false;
    ie::Inkling2Result res;
    std::vector<float> rec;
    auto write_acts = [&] {
        if (acts.is_open() && acts_mean && acts_last > 1) {             // the mean over the recorded positions
            const size_t per = rec.size() / acts_last;
            std::vector<float> mean(per, 0.f);
            for (uint32_t r = 0; r < acts_last; ++r)
                for (size_t i = 0; i < per; ++i) mean[i] += rec[r * per + i] / float(acts_last);
            acts.write(reinterpret_cast<const char*>(mean.data()), std::streamsize(mean.size() * 4));
        } else if (acts.is_open()) acts.write(reinterpret_cast<const char*>(rec.data()), std::streamsize(rec.size() * 4));
    };
    // one prompt's json line (the reply as the server parses it: answer text, reasoning; the raw ids for the degenerate check)
    auto reply_line = [&](const std::string& cand, size_t pi, const std::vector<int32_t>& out, bool ended, const json& kl) {
        ie::Inkling2ReplyParser parser(true);
        for (const auto& [mk, txt] : ie::inkling2_events_from_text(tok.decode(out))) parser.feed(mk, txt);
        parser.finish(ended);
        json r;
        r["candidate"] = cand; r["name"] = P[pi].name; r["ids"] = out; r["ended"] = ended;
        r["text"] = parser.content(); r["reasoning"] = parser.reasoning(); r["kl"] = kl;
        r["tool_calls"] = json::parse(parser.tool_calls_json(), nullptr, false);
        return r.dump(-1, ' ', false, json::error_handler_t::replace);
    };
    if (parallel) {                                                     // the slots' state, once for every candidate
        size_t mx = 0;
        for (const auto& p : P) mx = std::max(mx, p.ids.size());
        const uint32_t B = uint32_t(std::min<size_t>(parallel, P.size()));
        const uint32_t ctx = uint32_t(mx) + std::max(gen_n, kl_steps) + 1;
        if (auto e = m.batch_begin(B, ctx); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
        std::printf("parallel: %u slots of %u positions, %.2f GiB on the card in all\n", B, ctx, m.device_bytes() / 1073741824.0);
        std::fflush(stdout);
    }
    for (size_t ci = 0; ci < cands.size(); ++ci) {
        const Cand& cd = cands[ci];
        if (!ablate_list.empty()) {
            if (cd.path == "none" || cd.path.empty()) { if (auto e = m.set_ablation(nullptr); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; } }
            else {
                ie::Inkling2Ablation a;
                if (auto e = ie::inkling2_read_ablation(cd.path, cfg, a); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
                if (auto e = m.set_ablation(&a); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
            }
        }
        const auto t0 = std::chrono::steady_clock::now();
        uint64_t ntok = 0, nended = 0;
        double kl_sum = 0; uint64_t kl_n = 0;
        if (parallel) {
            // Every prompt runs as in the sequential loop below, its decode steps shared with the other slots' (one
            // batch_step per round); the json lines are held back and written in prompt order.
            const uint32_t B = uint32_t(std::min<size_t>(parallel, P.size()));
            const size_t V = cfg.vocab_unpadded;
            struct Conv {
                int64_t pi = -1;
                int phase = 0;                                          // 1 reply, 2 teacher-forced KL
                std::vector<int32_t> out;
                bool ended = false, keep_ref = false;
                std::vector<std::vector<float>> lg;
                json kl = json::array();
                int32_t next = -1;                                      // the id to feed at the next step (-1: none)
            };
            std::vector<Conv> cv(B);
            std::vector<std::string> lines(P.size());
            std::vector<char> ready(P.size(), 0);
            size_t next_prompt = 0, flushed = 0;
            const bool want_gen = gens.is_open() && gen_n;
            std::string err;
            auto done = [&](Conv& c, bool has_line) {
                if (has_line) { lines[size_t(c.pi)] = reply_line(cd.name, size_t(c.pi), c.out, c.ended, c.kl); }
                ready[size_t(c.pi)] = 1;
                while (flushed < P.size() && ready[flushed]) {
                    if (!lines[flushed].empty()) { gens << lines[flushed] << "\n"; gens.flush(); }
                    lines[flushed].clear();
                    ++flushed;
                }
                c = Conv{};
            };
            auto kl_logits_row = [&](Conv& c, uint32_t slot, const float* lg) {     // a teacher-forced KL position
                const size_t pi = size_t(c.pi), k = c.kl.size();
                const double d = kl_logits(first_lg[pi][k], std::vector<float>(lg, lg + V));
                c.kl.push_back(d); kl_sum += d; ++kl_n;
                if (k + 1 == first_lg[pi].size()) { done(c, true); return; }
                c.next = first_ids[pi][k];
                (void)slot;
            };
            auto end_reply = [&](Conv& c, uint32_t slot) {
                const size_t pi = size_t(c.pi);
                ntok += c.out.size(); nended += c.ended;
                if (c.keep_ref) { first_ids[pi] = c.out; first_lg[pi] = std::move(c.lg); }
                else if ((ci > 0 || ref_loaded) && kl_steps && !first_lg[pi].empty()) {   // teacher-force the first candidate's reply
                    ie::Inkling2Result r2;
                    if (auto e = m.batch_prefill(slot, P[pi].ids, ro, r2); !e.empty()) { err = e; return; }
                    c.phase = 2;
                    kl_logits_row(c, slot, r2.last_logits.data());
                    return;
                }
                done(c, true);
            };
            auto reply_row = [&](Conv& c, uint32_t slot, const float* lg) {         // one reply position
                const size_t k = c.out.size();
                if (c.keep_ref && k < kl_steps) c.lg.emplace_back(lg, lg + V);
                const int32_t id = argmax_row(lg, V);
                c.out.push_back(id);
                if (std::find(stop.begin(), stop.end(), id) != stop.end()) { c.ended = true; end_reply(c, slot); return; }
                if (c.out.size() == gen_n) { end_reply(c, slot); return; }
                c.next = id;
            };
            std::vector<uint32_t> rs;
            std::vector<int32_t> rid;
            std::vector<float> lgs;
            while (err.empty()) {
                for (uint32_t b = 0; b < B && next_prompt < P.size() && err.empty(); ++b) {     // admit in prompt order
                    if (cv[b].pi >= 0) continue;
                    Conv& c = cv[b];
                    c.pi = int64_t(next_prompt++);
                    ro.record_last = acts.is_open() ? acts_last : 0;
                    ro.record_out = acts.is_open() ? &rec : nullptr;
                    ie::Inkling2Result r2;
                    const std::string e = m.batch_prefill(b, P[size_t(c.pi)].ids, ro, r2);
                    ro.record_last = 0; ro.record_out = nullptr;
                    if (!e.empty()) { err = P[size_t(c.pi)].name + ": " + e; break; }
                    write_acts();
                    if (!want_gen) { done(c, false); continue; }
                    c.phase = 1;
                    c.keep_ref = ci == 0 && kl_steps && !ref_loaded;
                    reply_row(c, b, r2.last_logits.data());
                }
                if (!err.empty()) break;
                rs.clear(); rid.clear();
                for (uint32_t b = 0; b < B; ++b) if (cv[b].pi >= 0 && cv[b].next >= 0) { rs.push_back(b); rid.push_back(cv[b].next); }
                if (rs.empty()) {
                    if (next_prompt >= P.size()) break;
                    continue;
                }
                if (auto e = m.batch_step(rs, rid, lgs); !e.empty()) { err = e; break; }
                for (size_t i = 0; i < rs.size() && err.empty(); ++i) {
                    Conv& c = cv[rs[i]];
                    c.next = -1;
                    if (c.phase == 1) reply_row(c, rs[i], lgs.data() + i * V);
                    else kl_logits_row(c, rs[i], lgs.data() + i * V);
                }
            }
            if (!err.empty()) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        }
        for (size_t pi = 0; pi < P.size() && !parallel; ++pi) {
            m.reset_state();
            ro.record_last = acts.is_open() ? acts_last : 0;
            ro.record_out = acts.is_open() ? &rec : nullptr;
            if (auto e = m.forward(P[pi].ids, ro, res); !e.empty()) { std::fprintf(stderr, "%s: %s\n", P[pi].name.c_str(), e.c_str()); return 1; }
            ro.record_last = 0; ro.record_out = nullptr;
            write_acts();
            if (!gens.is_open() || !gen_n) continue;
            std::vector<int32_t> out;
            bool ended = false;
            std::vector<std::vector<float>> lg_steps;
            const bool keep_ref = ci == 0 && kl_steps && !ref_loaded;
            for (uint32_t k = 0; k < gen_n; ++k) {
                if (keep_ref && k < kl_steps) lg_steps.push_back(res.last_logits);
                const int32_t id = top_n(res.last_logits, 1)[0];
                out.push_back(id);
                if (std::find(stop.begin(), stop.end(), id) != stop.end()) { ended = true; break; }
                if (k + 1 == gen_n) break;
                if (auto e = m.forward({id}, ro, res); !e.empty()) { std::fprintf(stderr, "%s: %s\n", P[pi].name.c_str(), e.c_str()); return 1; }
            }
            ntok += out.size(); nended += ended;
            json kl = json::array();
            if (keep_ref) { first_ids[pi] = out; first_lg[pi] = std::move(lg_steps); }
            else if ((ci > 0 || ref_loaded) && kl_steps && !first_lg[pi].empty()) {        // teacher-force the first candidate's reply
                m.reset_state();                                            // the prompt again (short; exact)
                if (auto e = m.forward(P[pi].ids, ro, res); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
                for (size_t k = 0; k < first_lg[pi].size(); ++k) {
                    const double d = kl_logits(first_lg[pi][k], res.last_logits);
                    kl.push_back(d); kl_sum += d; ++kl_n;
                    if (k + 1 == first_lg[pi].size()) break;
                    if (auto e = m.forward({first_ids[pi][k]}, ro, res); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
                }
            }
            gens << reply_line(cd.name, pi, out, ended, kl) << "\n";
            gens.flush();
        }
        if (ci == 0 && !ref_loaded && !kl_ref_out.empty()) {
            std::ofstream f(kl_ref_out, std::ios::binary);
            const uint32_t np = uint32_t(P.size()), vocab = cfg.vocab_unpadded;
            f.write(reinterpret_cast<const char*>(&np), 4); f.write(reinterpret_cast<const char*>(&vocab), 4);
            for (size_t pi = 0; pi < P.size(); ++pi) {
                const uint32_t n = uint32_t(first_ids[pi].size()), st = uint32_t(first_lg[pi].size());
                f.write(reinterpret_cast<const char*>(&n), 4); f.write(reinterpret_cast<const char*>(first_ids[pi].data()), n * 4);
                f.write(reinterpret_cast<const char*>(&st), 4);
                for (const auto& v : first_lg[pi]) f.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size()) * 4);
            }
        }
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("candidate %-32s %zu prompts, %llu reply tokens (%llu turns ended), %.1f s%s\n", cd.name.c_str(), P.size(),
                    (unsigned long long)ntok, (unsigned long long)nended, sec,
                    kl_n ? (", mean KL vs the first candidate " + std::to_string(kl_sum / kl_n)).c_str() : "");
        std::fflush(stdout);
    }
    return 0;
}

static void print_profile(ie::KernelProfiler& prof, uint32_t steps, double dec) {
    auto st = prof.harvest();
    std::sort(st.begin(), st.end(), [](const auto& a, const auto& b) { return a.total_ns > b.total_ns; });
    uint64_t busy = 0, launches = 0;
    for (const auto& x : st) { busy += x.total_ns; launches += x.calls; }
    std::printf("profile over %u decode steps: GPU kernel time %.2f ms/step, wall %.2f ms/step (%.0f %% of the wall in kernels)\n",
                steps, busy * 1e-6 / steps, dec * 1e3 / steps, 100.0 * busy * 1e-9 / dec);
    static const size_t nshow = [] { const char* e = std::getenv("IE_INKLING2_PROFILE_ROWS"); return e ? size_t(std::atoi(e)) : size_t(14); }();
    for (size_t i = 0; i < std::min<size_t>(st.size(), nshow); ++i)
        std::printf("  %-22s %6u launches/step  %8.3f ms/step  (%4.1f %%)\n", st[i].name.c_str(), st[i].calls / steps,
                    st[i].total_ms() / steps, 100.0 * st[i].total_ns / busy);
    std::printf("  kernel launches per step: %llu\n", (unsigned long long)(launches / steps));
}

int main(int argc, char** argv) {
    std::string sample_check;
    std::string model = std::string(std::getenv("HOME") ? std::getenv("HOME") : ".") + "/models/Inkling-2-Mini", prompt, routes_path, teacher_dir, dump_dir, ids_from,
                step_routes_path, force_ids_arg, dump_steps, dump_features, draft_check, draft_gold, image_path, dump_vision, teacher_steps, bench, bench_prefill, bench_rows, bench_batch,
                bench_ctx = "500,3000";
    std::vector<int32_t> ids;
    uint32_t spec_k = 0, card = 0, repeat = 1, generate = 0, chunk = 0, chunk2 = 0, max_chunk = 256, ctx = 0, equiv = 0, top_k = 0,
             trunc_check = 0, rollback = 0;
    float temp = 0.f, top_p = 1.f;
    uint64_t seed = 42;
    bool ref_bf16 = false, tokenize_only = false, have_prompt = false, mem_table = false, profile = false, tp = false, spec = false, bench_real = false;
    std::string ablation, batch, effort, acts_out, gen_out, ablate_list, kl_ref_in, kl_ref_out;
    uint32_t parallel = 0;
    uint32_t acts_last = 1, gen_n = 0, kl_steps = 16;
    bool open_turn = false, acts_mean = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return argv[++i]; };
        auto u32 = [&]() { return uint32_t(std::strtoul(next().c_str(), nullptr, 10)); };
        if (a == "--model") model = next();
        else if (a == "--ids") ids = parse_ids(next());
        else if (a == "--ids-from") ids_from = next();
        else if (a == "--prompt") { prompt = next(); have_prompt = true; }
        else if (a == "--card") card = u32();
        else if (a == "--routes") routes_path = next();
        else if (a == "--teacher") teacher_dir = next();
        else if (a == "--dump-calls") dump_dir = next();
        else if (a == "--repeat") repeat = std::max(1u, u32());
        else if (a == "--ref-bf16") ref_bf16 = true;
        else if (a == "--tokenize-only") tokenize_only = true;
        else if (a == "--generate") generate = u32();
        else if (a == "--temp") temp = std::stof(next());
        else if (a == "--top-k") top_k = u32();
        else if (a == "--top-p") top_p = std::stof(next());
        else if (a == "--seed") seed = std::stoull(next());
        else if (a == "--step-routes") step_routes_path = next();
        else if (a == "--force-ids") force_ids_arg = next();
        else if (a == "--dump-steps") dump_steps = next();
        else if (a == "--dump-features") dump_features = next();
        else if (a == "--draft-check") { draft_check = next(); draft_gold = next(); }
        else if (a == "--spec") spec = true;
        else if (a == "--spec-k") spec_k = u32();
        else if (a == "--teacher-steps") teacher_steps = next();
        else if (a == "--chunk") chunk = u32();
        else if (a == "--chunk2") chunk2 = u32();
        else if (a == "--max-chunk") max_chunk = u32();
        else if (a == "--sample-check") sample_check = next();
        else if (a == "--ctx") ctx = u32();
        else if (a == "--equiv") equiv = u32();
        else if (a == "--mem-table") mem_table = true;
        else if (a == "--profile") profile = true;
        else if (a == "--tp") tp = true;
        else if (a == "--image") image_path = next();
        else if (a == "--dump-vision") dump_vision = next();
        else if (a == "--truncate-check") trunc_check = u32();
        else if (a == "--rollback") rollback = u32();
        else if (a == "--bench") bench = next();
        else if (a == "--bench-prefill") bench_prefill = next();
        else if (a == "--bench-rows") bench_rows = next();
        else if (a == "--bench-batch") bench_batch = next();
        else if (a == "--bench-real") bench_real = true;
        else if (a == "--bench-ctx") bench_ctx = next();
        else if (a == "--ablation") ablation = next();
        else if (a == "--batch") batch = next();
        else if (a == "--effort") effort = next();
        else if (a == "--open-turn") open_turn = true;
        else if (a == "--acts-out") acts_out = next();
        else if (a == "--acts-last") acts_last = std::max(1u, u32());
        else if (a == "--acts-mean") acts_mean = true;
        else if (a == "--gen") gen_n = u32();
        else if (a == "--gen-out") gen_out = next();
        else if (a == "--kl-steps") kl_steps = u32();
        else if (a == "--kl-ref-in") kl_ref_in = next();
        else if (a == "--kl-ref-out") kl_ref_out = next();
        else if (a == "--ablate-list") ablate_list = next();
        else if (a == "--parallel") parallel = u32();
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    ie::Tokenizer tok;
    const bool want_tok = have_prompt || generate > 0 || !batch.empty();
    if (want_tok) {
        if (auto e = tok.load_from_hf_json(model + "/tokenizer.json", model + "/tokenizer_config.json"); !e.empty()) {
            std::fprintf(stderr, "tokenizer: %s\n", e.c_str()); return 1;
        }
    }
    if (have_prompt) {
        ids = tok.encode(prompt, false);
        std::printf("prompt ids:");
        for (int32_t t : ids) std::printf(" %d", t);
        std::printf("\n");
        if (tokenize_only) return 0;
    }
    json meta2;
    if (!ids_from.empty()) {
        meta2 = load_json(ids_from);
        ids = meta2.contains("prompt_ids") ? meta2["prompt_ids"].get<std::vector<int32_t>>() : meta2["ids"].get<std::vector<int32_t>>();
    }
    std::vector<BatchPrompt> bprompts;
    if (!batch.empty()) {                                               // render every prompt as the server does
        std::ifstream bf(batch);
        if (!bf) { std::fprintf(stderr, "cannot open %s\n", batch.c_str()); return 1; }
        const bool thinking = effort != "off";
        const std::string eff = (effort == "off" || effort == "default") ? std::string() : effort;
        std::string line;
        size_t mn = SIZE_MAX, mx = 0;
        while (std::getline(bf, line)) {
            if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
            const json j = json::parse(line);
            ie::ChatTurn u;
            u.role = "user";
            u.content = j.at("text").get<std::string>();
            std::string err;
            std::vector<std::pair<size_t, size_t>> content;
            const std::string pr = ie::inkling2_render_prompt(std::span<const ie::ChatTurn>(&u, 1), "", thinking, eff, err, &content, open_turn);
            if (!err.empty()) { std::fprintf(stderr, "render: %s\n", err.c_str()); return 1; }
            bprompts.push_back({j.at("name").get<std::string>(), tok.encode(pr, true, content)});
            if (j.contains("append_ids"))                               // e.g. the first tokens of a recorded reply
                for (const auto& v : j["append_ids"]) bprompts.back().ids.push_back(v.get<int32_t>());
            mn = std::min(mn, bprompts.back().ids.size()); mx = std::max(mx, bprompts.back().ids.size());
        }
        if (bprompts.empty()) { std::fprintf(stderr, "%s: no prompts\n", batch.c_str()); return 1; }
        std::string eff_text, eerr;
        ie::inkling2_effort_text(eff, thinking, eff_text, eerr);
        std::printf("batch: %zu prompts, %zu..%zu tokens, \"Thinking effort level: %s\"%s\n", bprompts.size(), mn, mx,
                    eff_text.c_str(), open_turn ? ", <|message_model|> appended" : "");
        ids = bprompts[0].ids;
    }
    if (ids.empty()) { std::fprintf(stderr, "need --ids, --ids-from or --prompt\n"); return 2; }

    ie::Inkling2Config cfg;
    if (auto e = ie::inkling2_read_config(model, cfg); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
    const uint32_t T = uint32_t(ids.size()), H = cfg.hidden, NC = cfg.n_calls();
    const auto sched = ie::inkling2_schedule(cfg);

    std::string err;
    std::vector<std::vector<std::array<int32_t, 8>>> routes;
    if (!routes_path.empty()) {
        routes = parse_routes(load_json(routes_path), NC, T, err);
        if (!err.empty()) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    }
    json step_routes;
    if (!step_routes_path.empty()) {
        step_routes = load_json(step_routes_path);
        if (!step_routes.contains("step0")) { std::fprintf(stderr, "step routes: no step0\n"); return 1; }
        routes = parse_routes(step_routes["step0"], NC, T, err);
        if (!err.empty()) { std::fprintf(stderr, "step0 %s\n", err.c_str()); return 1; }
    }
    std::vector<int32_t> force_ids;
    if (force_ids_arg == "meta") force_ids = meta2.at("generated").get<std::vector<int32_t>>();
    else if (!force_ids_arg.empty()) force_ids = parse_ids(force_ids_arg);
    std::vector<std::vector<float>> teacher;
    if (!teacher_dir.empty()) {
        for (uint32_t i = 0; i < NC; ++i) {
            const std::string p = golden_file(teacher_dir, int(i));
            if (p.empty()) { std::fprintf(stderr, "teacher: no call%02d file in %s\n", i, teacher_dir.c_str()); return 1; }
            std::vector<float> v;
            std::vector<int64_t> sh;
            if (auto e = read_npy_f32(p, v, sh); !e.empty()) { std::fprintf(stderr, "teacher: %s\n", e.c_str()); return 1; }
            if (v.size() != size_t(T) * H) { std::fprintf(stderr, "teacher: %s has %zu values, need %u x %u\n", p.c_str(), v.size(), T, H); return 1; }
            teacher.push_back(std::move(v));
        }
    }

    std::vector<sycl::device> arcs;
    for (const auto& pl : sycl::platform::get_platforms()) {
        if (pl.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : pl.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) arcs.push_back(d);
    }
    if (card >= arcs.size()) { std::fprintf(stderr, "no Arc GPU #%u\n", card); return 1; }
    sycl::property_list props = profile ? sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}
                                        : sycl::property_list{sycl::property::queue::in_order{}};
    if (tp && (arcs.size() < 2 || profile)) { std::fprintf(stderr, "--tp needs two cards and no --profile\n"); return 2; }
    const sycl::context qctx = tp ? sycl::context(std::vector<sycl::device>{arcs[card], arcs[1 - card]}) : sycl::context(arcs[card]);
    sycl::queue q(qctx, arcs[card], props);
    std::unique_ptr<sycl::queue> q2;
    if (tp) {
        q2 = std::make_unique<sycl::queue>(qctx, arcs[1 - card], props);
        std::printf("two cards on one conversation: card %u and card %u\n", card, 1 - card);
    }
    std::printf("device: %s (card %u)\nmode: %s%s%s\n", q.get_device().get_info<sycl::info::device::name>().c_str(), card,
                ref_bf16 ? "reference bf16 rounding emulated" : "fp32 (engine arithmetic)",
                routes.empty() ? "" : ", expert picks imposed", teacher.empty() ? "" : ", teacher-forced");

    if (!sample_check.empty()) {   // i2_sample_rows against Ds41Generator::sample_row / distribution on one logits row
        std::ifstream f(sample_check, std::ios::binary);
        std::vector<char> raw((std::istreambuf_iterator<char>(f)), {});
        std::vector<float> row(raw.size() / 4);
        std::memcpy(row.data(), raw.data(), row.size() * 4);
        const uint32_t V = std::min<uint32_t>(uint32_t(row.size()), 200058u);
        row.resize(V);
        struct Cfg { float t, tp, mp, rep; uint32_t tk; const char* name; };
        const Cfg cfgs[] = {{0.f, 1.f, 0.f, 1.f, 0, "greedy"}, {1.0f, 1.f, 0.f, 1.f, 0, "T 1.0"}, {1.0f, 0.95f, 0.f, 1.f, 0, "T 1.0 top_p 0.95"},
                            {0.7f, 1.f, 0.f, 1.f, 0, "T 0.7"}, {1.0f, 1.f, 0.f, 1.f, 40, "T 1.0 top_k 40"}, {1.0f, 1.f, 0.05f, 1.f, 0, "T 1.0 min_p 0.05"}, {0.7f, 0.9f, 0.05f, 1.f, 40, "T 0.7 top_k 40 top_p 0.9 min_p 0.05"}, {1.0f, 1.f, 0.f, 1.3f, 0, "T 1.0 repeat 1.3"}};
        const uint32_t N = std::getenv("IE_SC_N") ? uint32_t(std::atoi(std::getenv("IE_SC_N"))) : 4000u, B = 64;
        const bool nohost = std::getenv("IE_SC_NOHOST") != nullptr;
        const uint64_t seed0 = std::getenv("IE_SC_SEED") ? std::strtoull(std::getenv("IE_SC_SEED"), nullptr, 10) : 777ull;
        std::vector<int32_t> recent;                            // the 8 most likely ids, for the repetition penalty
        { std::vector<int32_t> ix(V); std::iota(ix.begin(), ix.end(), 0);
          std::partial_sort(ix.begin(), ix.begin() + 8, ix.end(), [&](int a, int b) { return row[a] > row[b]; }); recent.assign(ix.begin(), ix.begin() + 8); }
        float* dl = sycl::malloc_device<float>(uint64_t(B) * V, q);
        int32_t* dc = sycl::malloc_device<int32_t>(uint64_t(B) * V, q);
        int32_t* dout = sycl::malloc_device<int32_t>(B, q);
        ie::I2SampleRow* dr = sycl::malloc_device<ie::I2SampleRow>(B, q);
        int32_t* drec = sycl::malloc_device<int32_t>(8, q);
        q.memcpy(drec, recent.data(), 32).wait();
        bool all_ok = true;
        for (const Cfg& cf : cfgs) {
            ie::Ds41SampleParams sp; sp.temperature = cf.t; sp.top_p = cf.tp; sp.min_p = cf.mp; sp.top_k = cf.tk;
            sp.repeat_penalty = cf.rep; sp.repeat_window = cf.rep != 1.f ? 8 : 0;
            std::vector<int32_t> ids; std::vector<double> probs;
            { std::vector<float> tmp = row; ie::Ds41Generator::distribution(tmp.data(), V, recent, sp, ids, probs); }
            std::map<int32_t, double> exact; double zz = 0; for (double p0 : probs) zz += p0;
            for (size_t i = 0; i < ids.size(); ++i) exact[ids[i]] += probs[i] / zz;
            std::map<int32_t, double> hd, dd;
            uint64_t rng = 12345;
            for (uint32_t k = 0; k < (nohost ? 0u : N); ++k) { std::vector<float> tmp = row; hd[ie::Ds41Generator::sample_row(tmp.data(), V, recent, sp, rng)] += 1.0 / N; }
            uint64_t seed = seed0;
            for (uint32_t k0 = 0; k0 < N; k0 += B) {
                const uint32_t n = std::min(B, N - k0);
                std::vector<ie::I2SampleRow> rr(n);
                for (uint32_t i = 0; i < n; ++i) {
                    rr[i].temp = cf.t; rr[i].top_p = cf.tp; rr[i].min_p = cf.mp; rr[i].top_k = cf.tk; rr[i].rep = cf.rep;
                    rr[i].rec_off = 0; rr[i].rec_n = cf.rep != 1.f ? 8 : 0;
                    seed += 0x9E3779B97F4A7C15ull; uint64_t z = seed; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
                    rr[i].seed = z ^ (z >> 31);
                    q.memcpy(dl + uint64_t(i) * V, row.data(), uint64_t(V) * 4);
                }
                q.memcpy(dr, rr.data(), n * sizeof(ie::I2SampleRow));
                ie::i2_sample_rows(q, dl, n, V, dr, drec, nullptr, nullptr, dc, dout);
                std::vector<int32_t> o(n);
                q.memcpy(o.data(), dout, n * 4).wait();
                for (int32_t v : o) dd[v] += 1.0 / N;
            }
            auto tv = [&](const std::map<int32_t, double>& a) {
                std::map<int32_t, double> u = exact; double s2 = 0;
                for (auto& [k, v] : a) u[k] += 0;
                for (auto& [k, v] : u) { const auto it = a.find(k); s2 += std::fabs((it == a.end() ? 0.0 : it->second) - v); }
                return 0.5 * s2;
            };
            // the sampling noise of N draws: E[TV] ~ 0.5 * sum sqrt(2 p (1 - p) / (pi N))
            double noise = 0; for (auto& [k, v] : exact) noise += 0.5 * std::sqrt(2.0 * v * (1 - v) / (3.14159265 * N));
            const double th = tv(hd), td = tv(dd);
            int32_t top = -1; double tp0 = 0; for (auto& [k, v] : exact) if (v > tp0) { tp0 = v; top = k; }
            const bool ok = cf.t <= 0.f ? (dd.size() == 1 && dd.begin()->first == hd.begin()->first) : td < 2.0 * noise + 0.01;
            all_ok = all_ok && ok;
            std::printf("%-40s kept %6zu  TV(host, exact) %.4f  TV(card, exact) %.4f  (noise ~%.4f)  top %d: exact %.4f host %.4f card %.4f  %s\n",
                        cf.name, exact.size(), th, td, noise, top, tp0, hd[top], dd[top], ok ? "OK" : "DIFFERENT");
        }
        sycl::free(dl, q); sycl::free(dc, q); sycl::free(dout, q); sycl::free(dr, q); sycl::free(drec, q);
        std::printf("sample-check: %s\n", all_ok ? "PASS" : "FAIL");
        return all_ok ? 0 : 1;
    }

    ie::Inkling2Model m;
    auto t0 = std::chrono::steady_clock::now();
    ie::Inkling2LoadOptions lo;
    lo.ref_bf16 = ref_bf16;
    lo.max_chunk = max_chunk;
    lo.rollback = rollback;
    lo.ablation = !ablate_list.empty() ? std::string("off") : ablation;
    lo.tp_queue = q2.get();
    if (!batch.empty() && gen_n) lo.max_chunk = std::max<uint32_t>(lo.max_chunk, 256);
    if (auto e = m.load(model, q, lo); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
    std::printf("ablation: %s\n", m.ablation_summary().empty() ? "none" : m.ablation_summary().c_str());
    std::printf("%s\n", m.kv_note().c_str());
    const size_t after_load = m.device_bytes();
    std::printf("loaded in %.1f s, %.2f GiB on the device (weights + rings + conv states + scratch for chunks of %u)\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), after_load / 1073741824.0, max_chunk);
    if (ctx) { if (auto e = m.reserve(ctx); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; } }

    if (mem_table) {
        const double G = 1073741824.0;
        const size_t scr = ie::Inkling2Model::scratch_bytes_for(cfg, max_chunk, false);
        const size_t st0 = ie::Inkling2Model::state_bytes_for(cfg, 0, max_chunk);
        const size_t weights = after_load - scr - st0;
        const size_t per_tok = ie::Inkling2Model::state_bytes_for(cfg, 1, max_chunk) - st0;
        const size_t total = q.get_device().get_info<sycl::info::device::global_mem_size>();
        std::printf("memory (own accounting, fp32 K/V, max chunk %u): weights %.2f GiB, scratch %.2f GiB, rings + conv states %.2f GiB, "
                    "%.1f KiB per position of context (8 global entries)\n", max_chunk, weights / G, scr / G, st0 / G, per_tok / 1024.0);
        for (uint32_t c : {8192u, 32768u, 131072u})
            std::printf("  context %6u: state %.2f GiB, total %.2f GiB\n", c, ie::Inkling2Model::state_bytes_for(cfg, c, max_chunk) / G,
                        (weights + scr + ie::Inkling2Model::state_bytes_for(cfg, c, max_chunk)) / G);
        std::printf("  device global memory %.2f GiB -> largest context by this accounting: %zu positions (no headroom left)\n",
                    total / G, (total - weights - scr - st0) / per_tok);
    }

    std::vector<float> img_rows;
    std::vector<uint32_t> img_at;
    ie::Inkling2Vision vis;
    if (!image_path.empty()) {
        std::ifstream f(image_path, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        ie::Inkling2Image img;
        if (auto e = ie::inkling2_load_image(bytes.data(), bytes.size(), img); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
        for (uint32_t i = 0; i < T; ++i) if (ids[i] == 200054) img_at.push_back(i);
        if (img_at.size() != img.tokens()) {
            std::fprintf(stderr, "image: %u x %u tokens, the prompt has %zu placeholders\n", img.rows, img.cols, img_at.size());
            return 1;
        }
        auto tv = std::chrono::steady_clock::now();
        if (auto e = vis.init(q, model, std::max<uint32_t>(img.tokens(), 64)); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
        std::vector<std::vector<float>> capt;
        auto te = std::chrono::steady_clock::now();
        if (auto e = vis.encode(img, img_rows, dump_vision.empty() ? nullptr : &capt); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
        std::printf("image: %u x %u tokens; tower loaded in %.1f s (%.2f GiB), encoded in %.3f s\n", img.rows, img.cols,
                    std::chrono::duration<double>(te - tv).count(), vis.device_bytes() / 1073741824.0,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - te).count());
        if (!dump_vision.empty()) {
            fs::create_directories(dump_vision);
            const char* names[3] = {"/tower_embed.f32", "/tower_layer0.f32", "/tower_last.f32"};
            for (size_t i = 0; i < capt.size() && i < 3; ++i) write_raw(dump_vision + names[i], capt[i].data(), capt[i].size());
            write_raw(dump_vision + "/image_features.f32", img_rows.data(), img_rows.size());
        }
    }

    ie::Inkling2RunOptions ro;
    if (!img_at.empty()) { ro.image_rows = &img_rows; ro.image_at = &img_at; }
    ro.forced_routes = routes.empty() ? nullptr : &routes;
    ro.teacher = teacher.empty() ? nullptr : &teacher;
    ro.keep_calls = !dump_dir.empty();
    ro.chunk = chunk;
    ie::Inkling2Result res;
    auto secs = [](auto a) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count(); };

    if (!batch.empty()) return run_batch(m, tok, cfg, bprompts, ablate_list, acts_out, acts_last, gen_n, gen_out, kl_steps, acts_mean, kl_ref_in, kl_ref_out, parallel);

    if (!bench_batch.empty()) {                                         // batched decode throughput
        const std::vector<int32_t> cr = parse_ids(bench_ctx);
        if (cr.size() != 2 || cr[0] < 1 || cr[1] < cr[0]) { std::fprintf(stderr, "--bench-ctx needs LO,HI\n"); return 2; }
        for (int32_t B : parse_ids(bench_batch)) {
            if (B < 1) continue;
            if (auto e = m.batch_begin(uint32_t(B), uint32_t(cr[1]) + 64); !e.empty()) { std::fprintf(stderr, "B %d: %s\n", B, e.c_str()); return 1; }
            std::vector<uint32_t> sl(B);
            std::vector<int32_t> real_next(B, 0);
            for (int32_t b = 0; b < B; ++b) {
                sl[b] = uint32_t(b);
                const uint32_t n = B == 1 ? uint32_t(cr[0] + cr[1]) / 2 : uint32_t(cr[0] + int64_t(cr[1] - cr[0]) * b / (B - 1));
                if (bench_real) {                                       // a real context: a different stretch of the prompt each
                    const uint32_t at = uint32_t((uint64_t(b) * 977) % std::max<uint32_t>(1, T - n));
                    if (at + n > T) { std::fprintf(stderr, "--bench-real: the prompt is shorter than the context\n"); return 2; }
                    ie::Inkling2Result rr;
                    if (auto e = m.batch_forward(uint32_t(b), std::vector<int32_t>(ids.begin() + at, ids.begin() + at + n), ro, rr); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
                    real_next[b] = top_n(rr.last_logits, 1)[0];
                } else
                if (auto e = m.batch_fill_zero_for_benchmark(uint32_t(b), n); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
            }
            std::vector<int32_t> bid(B);
            std::vector<float> lg;
            auto step = [&](int k) {
                for (int32_t b = 0; b < B; ++b) bid[b] = bench_real ? real_next[b] : ids[(k + b) % T];
                if (auto e = m.batch_step(sl, bid, lg); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); std::exit(1); }
                if (bench_real)                                         // greedy: each row continues its own text
                    for (int32_t b = 0; b < B; ++b) {
                        std::vector<float> row(lg.begin() + size_t(b) * (lg.size() / B), lg.begin() + size_t(b + 1) * (lg.size() / B));
                        real_next[b] = top_n(row, 1)[0];
                    }
            };
            for (int w = 0; w < 3; ++w) step(w);
            ie::KernelProfiler prof;
            if (profile) { ie::g_profiler = &prof; prof.begin_step(); }
            const auto t1 = std::chrono::steady_clock::now();
            for (int k = 0; k < 16; ++k) { if (profile) prof.mark_token(); step(k); }
            const double sec = secs(t1);
            std::printf("BENCH batch %3d (contexts %d..%d): %8.1f tok/s in all, %6.2f tok/s per conversation (%.1f ms/step), "
                        "%.2f GiB on the card\n", B, B == 1 ? (cr[0] + cr[1]) / 2 : cr[0], B == 1 ? (cr[0] + cr[1]) / 2 : cr[1],
                        B * 16 / sec, 16 / sec, sec / 16 * 1e3, m.device_bytes() / 1073741824.0);
            std::fflush(stdout);
            if (profile) { q.wait(); ie::g_profiler = nullptr; print_profile(prof, 16, sec); }
            m.batch_end();
        }
        return 0;
    }

    if (!bench.empty() || !bench_prefill.empty()) {
        ro.record_routes = false;
        if (!bench_prefill.empty()) {                                   // warm-up: JIT every prefill / tail kernel first
            m.reset_state();
            const uint32_t w = std::min<uint32_t>(T, 300);
            if (auto e = m.forward(std::vector<int32_t>(ids.begin(), ids.begin() + w), ro, res); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
        }
        for (int32_t n : parse_ids(bench_prefill)) {
            if (uint32_t(n) > T) { std::fprintf(stderr, "--bench-prefill %d: prompt has %u ids\n", n, T); return 2; }
            m.reset_state();
            m.reserve(uint32_t(n) + 8);
            ie::KernelProfiler prof;
            if (profile) { ie::g_profiler = &prof; prof.begin_step(); prof.mark_token(); }
            const auto t1 = std::chrono::steady_clock::now();
            if (auto e = m.forward(std::vector<int32_t>(ids.begin(), ids.begin() + n), ro, res); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
            const double sec = secs(t1);
            std::printf("BENCH prefill %6d tokens: %.1f tok/s (%.2f s)\n", n, n / sec, sec);
            if (profile) { q.wait(); ie::g_profiler = nullptr; std::printf("(prefill profile: per-step = the whole prefill)\n"); print_profile(prof, 1, sec); }
        }
        for (int32_t n : parse_ids(bench)) {
            if (auto e = m.fill_zero_context_for_benchmark(uint32_t(n)); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
            m.reserve(uint32_t(n) + 32);
            std::vector<float> lg;
            for (int w = 0; w < 3; ++w) m.decode_step(ids[0], lg);
            ie::KernelProfiler prof;
            if (profile) { ie::g_profiler = &prof; prof.begin_step(); }
            const auto t1 = std::chrono::steady_clock::now();
            for (int k = 0; k < 16; ++k) {
                if (profile) prof.mark_token();
                if (auto e = m.decode_step(ids[k % T], lg); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
            }
            const double sec = secs(t1);
            std::printf("BENCH decode at context %6d: %.2f tok/s (%.1f ms/token)\n", n, 16 / sec, sec / 16 * 1e3);
            for (int32_t rows : parse_ids(bench_rows)) {               // a verify-sized forward, rewound each time
                const uint32_t base = m.n_pos();
                std::vector<int32_t> blk(ids.begin(), ids.begin() + std::min<uint32_t>(T, uint32_t(rows)));
                for (int w = 0; w < 2; ++w) {
                    if (auto e = m.forward(blk, ro, res); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
                    if (auto e = m.truncate(base); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
                }
                const auto t3 = std::chrono::steady_clock::now();
                for (int k = 0; k < 8; ++k) {
                    if (auto e = m.forward(blk, ro, res); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
                    if (auto e = m.truncate(base); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
                }
                std::printf("BENCH forward of %2zu rows at context %6d (+ truncate): %.2f ms\n", blk.size(), n, secs(t3) / 8 * 1e3);
            }
            if (profile) { q.wait(); ie::g_profiler = nullptr; print_profile(prof, 16, sec); }
        }
        std::printf("device memory in use: %.2f GiB\n", m.device_bytes() / 1073741824.0);
        return 0;
    }

    if (trunc_check) {                                                  // truncate() = never having fed the rewound tokens
        ro.record_routes = false;
        const uint32_t D = trunc_check;
        if (D >= T) { std::fprintf(stderr, "--truncate-check needs D below the prompt length\n"); return 2; }
        auto fw = [&](const std::vector<int32_t>& v) { if (auto e = m.forward(v, ro, res); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); std::exit(1); } };
        m.reset_state();
        fw(ids);
        const auto ref = res.last_logits;
        m.reset_state();
        fw(ids);
        std::vector<int32_t> junk(ids.rbegin(), ids.rbegin() + D);   // D different tokens after the prompt
        fw(junk);
        const uint32_t target = T - D;
        std::printf("fed %u + %u tokens; earliest restorable %u; truncate to %u\n", T, D, m.earliest_restorable(), target);
        if (auto e = m.truncate(target); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
        fw(std::vector<int32_t>(ids.begin() + target, ids.end()));
        double rel, mx;
        rel_err(res.last_logits, ref, rel, mx);
        const bool same = top_n(res.last_logits, 10) == top_n(ref, 10);
        const uint32_t too_far = m.earliest_restorable() ? m.earliest_restorable() - 1 : 0;
        const std::string refusal = too_far ? m.truncate(too_far) : std::string("(nothing below position 0)");
        std::printf("truncate-check: logits relL2 %.3e max|d| %.3e vs the straight run, top-10 %s; truncate(%u) below the earliest: %s\n",
                    rel, mx, same ? "identical" : "DIFFERENT", too_far, refusal.empty() ? "ACCEPTED (wrong)" : refusal.c_str());
        const bool ok = rel <= 1e-6 && same && (!too_far || !refusal.empty());
        std::printf("RESULT (truncate): %s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }

    if (equiv) {                                                        // 2b: cached vs uncached in the engine itself
        if (equiv >= T) { std::fprintf(stderr, "--equiv needs a split below the prompt length\n"); return 2; }
        ro.record_routes = false;
        std::vector<std::pair<std::string, std::vector<float>>> runs;
        auto one = [&](const std::string& name, auto&& body) {
            m.reset_state();
            const auto t1 = std::chrono::steady_clock::now();
            body();
            std::printf("  %-44s %.2f s\n", name.c_str(), secs(t1));
            runs.emplace_back(name, res.last_logits);
        };
        ro.chunk = T;
        if (T > max_chunk) { std::fprintf(stderr, "--equiv: the whole-prompt run needs --max-chunk >= %u\n", T); return 2; }
        one("whole prompt in one chunk", [&] { if (auto e = m.forward(ids, ro, res); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); std::exit(1); } });
        one("first " + std::to_string(equiv) + " at once, then one at a time", [&] {
            std::vector<int32_t> head(ids.begin(), ids.begin() + equiv);
            if (auto e = m.forward(head, ro, res); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); std::exit(1); }
            for (uint32_t t = equiv; t < T; ++t)
                if (auto e = m.forward({ids[t]}, ro, res); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); std::exit(1); }
        });
        for (uint32_t c : {chunk, chunk2}) {
            if (!c) continue;
            ro.chunk = c;
            one("chunks of " + std::to_string(c), [&] { if (auto e = m.forward(ids, ro, res); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); std::exit(1); } });
        }
        const auto ref_top = top_n(runs[0].second, 10);
        std::printf("against the whole-prompt run (top-1 %d):\n", ref_top[0]);
        for (size_t i = 1; i < runs.size(); ++i) {
            double rel, mx;
            rel_err(runs[i].second, runs[0].second, rel, mx);
            const auto tp = top_n(runs[i].second, 10);
            std::printf("  %-44s logits relL2 %.3e  max|d| %.3e  top-1 %d  top-10 %s\n", runs[i].first.c_str(), rel, mx, tp[0],
                        tp == ref_top ? "identical" : "DIFFERENT");
        }
        return 0;
    }

    std::unique_ptr<ie::Inkling2Draft> dr;
    if (!draft_check.empty() || spec) {
        dr = std::make_unique<ie::Inkling2Draft>();
        const auto td = std::chrono::steady_clock::now();
        if (auto e = dr->init(q, model, m.unembed_device(), cfg.vocab_unpadded, ctx ? ctx : 65536); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
        std::printf("draft head loaded in %.1f s, %.2f GiB\n", secs(td), dr->device_bytes() / 1073741824.0);
    }
    if (!draft_check.empty()) {                                         // the draft head against the CPU harness
        const json tj = json::parse(std::ifstream(draft_check + "/tokens.json"));
        const auto tids = tj.at("ids").get<std::vector<int32_t>>();
        const uint32_t P0 = tj.at("prompt_len"), taps = uint32_t(tj.at("taps").size());
        std::ifstream ff(draft_check + "/features.f32", std::ios::binary);
        ff.seekg(0, std::ios::end);
        const size_t nbytes = size_t(ff.tellg()); ff.seekg(0);
        const uint32_t N = uint32_t(nbytes / (size_t(taps) * H * 4));
        std::vector<float> rowmaj(size_t(N) * taps * H);
        ff.read(reinterpret_cast<char*>(rowmaj.data()), std::streamsize(nbytes));
        std::vector<float> tm(rowmaj.size());                           // [N][taps][H] -> [taps][N][H]
        for (uint32_t t = 0; t < N; ++t) for (uint32_t k = 0; k < taps; ++k)
            std::memcpy(tm.data() + (size_t(k) * N + t) * H, rowmaj.data() + (size_t(t) * taps + k) * H, H * 4);
        const json gold = json::parse(std::ifstream(draft_gold));
        std::map<uint32_t, std::vector<int32_t>> want;
        for (const auto& g : gold) if (fs::path(g.at("feat").get<std::string>()).filename() == fs::path(draft_check).filename())
            want[g.at("anchor").get<uint32_t>()] = g.at("drafts").get<std::vector<int32_t>>();
        uint32_t same_all = 0, same_first = 0, n_anchor = 0, acc = 0;
        double t_draft = 0;
        const uint32_t Bd = dr->block();
        if (auto e = dr->add_context(tm, N, 0, P0); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
        for (uint32_t pp = P0; pp + Bd <= tids.size(); ++pp) {
            if (pp > P0) if (auto e = dr->add_context(tm, N, pp - 1, 1); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
            std::vector<int32_t> d;
            const auto t3 = std::chrono::steady_clock::now();
            if (auto e = dr->draft(tids[pp], d); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
            t_draft += secs(t3);
            uint32_t a = 0;
            while (a < d.size() && d[a] == tids[pp + 1 + a]) ++a;
            acc += a; ++n_anchor;
            auto it = want.find(pp);
            if (it != want.end()) { same_all += it->second == d; same_first += it->second[0] == d[0]; }
        }
        std::printf("draft-check %s: %u anchors, engine mean accepted %.3f; drafts equal to the harness: all 7 at %u, the first at %u "
                    "(of %zu golden anchors); %.2f ms a draft\n", draft_check.c_str(), n_anchor, double(acc) / n_anchor, same_all, same_first,
                    want.size(), t_draft / n_anchor * 1e3);
        return 0;
    }
    if (generate && spec) {                                             // greedy with the draft head
        if (temp > 0.f) { std::fprintf(stderr, "--spec: greedy only here\n"); return 2; }
        if (!ctx) m.reserve(T + generate + 16);
        std::vector<uint32_t> fvis = dr->feature_visits();
        std::vector<float> feats;
        ro.feature_visits = &fvis; ro.features = &feats; ro.record_routes = false;
        auto t1 = std::chrono::steady_clock::now();
        if (auto e = m.forward(ids, ro, res); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
        ro.image_rows = nullptr; ro.image_at = nullptr;
        if (auto e = dr->add_context(feats, T, 0, T); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
        const double pre = secs(t1);
        std::ofstream sff;
        if (!dump_features.empty()) { fs::create_directories(dump_features); sff.open(dump_features + "/features.f32", std::ios::binary); }
        auto spec_feats = [&](uint32_t Tn, uint32_t rows) {           // committed rows, [T][taps][H]
            if (!sff) return;
            const size_t nt = fvis.size();
            for (uint32_t t = 0; t < rows; ++t) for (size_t k = 0; k < nt; ++k)
                sff.write(reinterpret_cast<const char*>(feats.data() + (k * Tn + t) * H), std::streamsize(H * 4));
        };
        spec_feats(T, T);
        std::vector<int32_t> out{top_n(res.last_logits, 1)[0]};
        std::vector<uint32_t> hist(dr->block(), 0);
        double t_d = 0, t_v = 0;
        uint32_t steps = 0;
        const auto t2 = std::chrono::steady_clock::now();
        ro.all_logits = true;
        while (out.size() < generate) {
            const int32_t anchor = out.back();
            std::vector<int32_t> d;
            const auto ta = std::chrono::steady_clock::now();
            if (auto e = dr->draft(anchor, d); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
            const auto tb = std::chrono::steady_clock::now();
            if (spec_k && spec_k < d.size()) d.resize(spec_k);
            std::vector<int32_t> blk{anchor};
            blk.insert(blk.end(), d.begin(), d.end());
            const uint32_t base = m.n_pos();
            if (auto e = m.forward(blk, ro, res); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
            t_d += std::chrono::duration<double>(tb - ta).count(); t_v += secs(tb);
            const uint32_t V = cfg.vocab_unpadded, Bn = uint32_t(blk.size());
            uint32_t a = 0;
            std::vector<int32_t> commit;
            for (uint32_t j = 0; j < Bn; ++j) {
                std::vector<float> row(res.all_logits.begin() + size_t(j) * V, res.all_logits.begin() + size_t(j + 1) * V);
                const int32_t t = top_n(row, 1)[0];
                commit.push_back(t);
                if (j + 1 < Bn && t == blk[j + 1]) ++a; else break;
            }
            ++hist[a]; ++steps;
            // keep positions base .. base + a (the anchor and the accepted drafts); the last commit is the next anchor
            if (a + 1 < Bn) if (auto e = m.truncate(base + a + 1); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
            if (auto e = dr->add_context(feats, Bn, 0, a + 1); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
            spec_feats(Bn, a + 1);
            for (int32_t t : commit) { if (out.size() < generate) out.push_back(t); }
        }
        const double dec = secs(t2);
        std::printf("generated ids:");
        for (int32_t t : out) std::printf(" %d", t);
        std::printf("\ntext: %s\n", json(tok.decode(out)).dump().c_str());
        double mean = 0; for (uint32_t k = 0; k < hist.size(); ++k) mean += double(k) * hist[k];
        std::printf("spec: %u verifies, mean accepted %.2f (tokens per verify %.2f), histogram", steps, mean / steps, 1 + mean / steps);
        for (uint32_t h : hist) std::printf(" %u", h);
        std::printf("\nprefill %.3f s; decode %zu tokens in %.3f s = %.2f tok/s (draft %.2f ms, verify %.2f ms per step); context %u\n",
                    pre, out.size() - 1, dec, (out.size() - 1) / dec, t_d / steps * 1e3, t_v / steps * 1e3, m.n_pos());
        return 0;
    }
    if (generate) {                                                     // prefill + one token at a time
        ie::Ds41SampleParams sp;
        sp.temperature = temp; sp.top_k = top_k; sp.top_p = top_p; sp.seed = seed;
        uint64_t rng = seed;
        if (!dump_steps.empty()) fs::create_directories(dump_steps);
        json steps = json::array();
        auto pick = [&](std::vector<float>& lg, const std::vector<int32_t>& recent) -> int32_t {
            if (temp <= 0.f) return top_n(lg, 1)[0];                    // greedy, lower id on exact ties
            return ie::Ds41Generator::sample(lg, recent, sp, rng);
        };
        auto record = [&](uint32_t k, const std::vector<int32_t>& in) {
            if (dump_steps.empty()) return;
            char name[32];
            std::snprintf(name, sizeof name, "/step%u.f32", k);
            write_raw(dump_steps + name, res.last_logits.data(), res.last_logits.size());
            json st;
            st["step"] = k; st["input_ids"] = in; st["own"] = json::array(); st["margin"] = json::array();
            for (uint32_t i = 0; i < NC; ++i) {
                const uint32_t n = uint32_t(res.own_routes[i].size() / 8);
                json o = json::array();
                for (uint32_t t = 0; t < n; ++t) o.push_back(std::vector<int32_t>(res.own_routes[i].begin() + t * 8, res.own_routes[i].begin() + t * 8 + 8));
                st["own"].push_back(o); st["margin"].push_back(res.route_margin[i]);
            }
            steps.push_back(st);
        };
        if (!ctx) m.reserve(T + generate);
        std::vector<std::vector<float>> tsteps;
        auto set_teacher = [&](uint32_t k, uint32_t Tk) -> bool {
            ro.teacher = nullptr; ro.keep_calls = false;
            if (teacher_steps.empty()) return true;
            char name[48];
            std::snprintf(name, sizeof name, "/hidden_step%u.npy", k);
            std::vector<float> v;
            std::vector<int64_t> sh;
            if (auto e = read_npy_f32(teacher_steps + name, v, sh); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return false; }
            if (v.size() != size_t(NC) * Tk * H) { std::fprintf(stderr, "%s: need [48, %u, %u]\n", name, Tk, H); return false; }
            tsteps.assign(NC, {});
            for (uint32_t i = 0; i < NC; ++i) tsteps[i].assign(v.begin() + size_t(i) * Tk * H, v.begin() + size_t(i + 1) * Tk * H);
            ro.teacher = &tsteps; ro.keep_calls = true;
            return true;
        };
        auto dump_calls = [&](uint32_t k) {
            if (!ro.keep_calls || dump_steps.empty()) return;
            char name[32];
            std::snprintf(name, sizeof name, "/calls%u.f32", k);
            std::ofstream f(dump_steps + name, std::ios::binary);
            for (const auto& c : res.calls) f.write(reinterpret_cast<const char*>(c.data()), std::streamsize(c.size() * 4));
        };
        std::vector<uint32_t> fvis;
        std::vector<float> feats;
        std::ofstream ffile;
        std::vector<int32_t> fed(ids.begin(), ids.end());
        if (!dump_features.empty()) {
            const json mc = json::parse(std::ifstream(model + "/config.json")).at("mtp_config");
            fvis = mc.at("target_layer_ids").get<std::vector<uint32_t>>();
            fs::create_directories(dump_features);
            ffile.open(dump_features + "/features.f32", std::ios::binary);
            ro.feature_visits = &fvis; ro.features = &feats;
        }
        auto write_feats = [&]() {                                      // [taps][T][H] -> [T][taps][H]
            if (fvis.empty()) return;
            const size_t nt = fvis.size(), Tn = feats.size() / (nt * H);
            for (size_t t = 0; t < Tn; ++t)
                for (size_t k = 0; k < nt; ++k)
                    ffile.write(reinterpret_cast<const char*>(feats.data() + (k * Tn + t) * H), std::streamsize(H * 4));
        };
        if (!set_teacher(0, T)) return 1;
        auto t1 = std::chrono::steady_clock::now();
        if (auto e = m.forward(ids, ro, res); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
        ro.image_rows = nullptr; ro.image_at = nullptr;                // the image rows belong to the prompt forward only
        const double pre = secs(t1);
        write_feats();
        record(0, {});
        dump_calls(0);
        std::vector<int32_t> out, recent(ids.end() - std::min<size_t>(ids.size(), 64), ids.end());
        std::vector<std::vector<std::array<int32_t, 8>>> sr;
        double dec = 0;
        ie::KernelProfiler prof;
        if (profile) { ie::g_profiler = &prof; prof.begin_step(); }
        for (uint32_t k = 1; k <= generate; ++k) {
            int32_t nxt = pick(res.last_logits, recent);
            out.push_back(nxt);
            recent.push_back(nxt);
            if (k == generate) break;                                   // the last pick needs no forward
            const int32_t feed = k - 1 < force_ids.size() ? force_ids[k - 1] : nxt;
            ro.forced_routes = nullptr;
            if (!step_routes.is_null()) {
                const std::string key = "step" + std::to_string(k);
                if (step_routes.contains(key)) {
                    sr = parse_routes(step_routes[key], NC, 1, err);
                    if (!err.empty()) { std::fprintf(stderr, "%s %s\n", key.c_str(), err.c_str()); return 1; }
                    ro.forced_routes = &sr;
                }
            }
            if (!set_teacher(k, 1)) return 1;
            if (profile) prof.mark_token();
            auto t2 = std::chrono::steady_clock::now();
            if (auto e = m.forward({feed}, ro, res); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
            dec += secs(t2);
            write_feats();
            fed.push_back(feed);
            record(k, {feed});
            dump_calls(k);
        }
        if (profile) { q.wait(); ie::g_profiler = nullptr; print_profile(prof, generate - 1, dec); }
        std::printf("generated ids:");
        for (int32_t t : out) std::printf(" %d", t);
        std::printf("\ntext: %s\n", json(tok.decode(out)).dump().c_str());
        std::printf("prefill: %u tokens in %.3f s = %.1f tok/s (first call includes JIT); decode: %u steps in %.3f s = %.2f tok/s; context %u\n",
                    T, pre, T / pre, generate - 1, dec, generate > 1 ? (generate - 1) / dec : 0.0, m.n_pos());
        std::printf("device memory in use: %.2f GiB\n", m.device_bytes() / 1073741824.0);
        if (!fvis.empty()) {
            fed.push_back(out.back());                                  // the last pick: an anchor, no features
            std::ofstream(dump_features + "/tokens.json") << json{{"ids", fed}, {"taps", fvis}, {"hidden", H}, {"prompt_len", ids.size()}}.dump();
        }
        if (!dump_steps.empty()) {
            json sj;
            sj["prompt_ids"] = ids; sj["generated"] = out; sj["forced_ids"] = force_ids; sj["steps"] = steps;
            std::ofstream(dump_steps + "/steps.json") << sj.dump();
        }
        return 0;
    }

    for (uint32_t r = 0; r < repeat; ++r) {
        if (r) m.reset_state();
        auto t1 = std::chrono::steady_clock::now();
        if (auto e = m.forward(ids, ro, res); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
        const double sec = secs(t1);
        std::printf("forward %u: %u tokens in %.3f s (%.1f tok/s)%s\n", r + 1, T, sec, T / sec, r == 0 ? " [includes JIT]" : "");
    }
    std::printf("device memory in use: %.2f GiB (weights + state + scratch)\n", m.device_bytes() / 1073741824.0);

    // own picks vs imposed
    if (!routes.empty()) {
        uint32_t ndiff = 0;
        for (uint32_t i = 0; i < NC; ++i)
            for (uint32_t t = 0; t < T; ++t)
                if (!std::equal(res.own_routes[i].begin() + t * 8, res.own_routes[i].begin() + t * 8 + 8, res.used_routes[i].begin() + t * 8)) {
                    if (ndiff++ < 20) std::printf("   own top-8 differs: call %2u layer %2u token %u: own 8th-9th margin %.2e\n", i, sched[i].layer, t, res.route_margin[i][t]);
                }
        std::printf("own top-8 differs from the imposed picks in %u of %u (call, token) pairs\n", ndiff, NC * T);
    }

    const auto& lg = res.last_logits;
    const auto order = top_n(lg, 10);
    std::printf("top-10 (last position):");
    for (int i = 0; i < 10; ++i) std::printf(" (%d, %.4f)", order[i], lg[order[i]]);
    std::printf("\n");
    if (have_prompt && tok.ready()) { const std::vector<int32_t> one{order[0]}; std::printf("top-1 text: '%s'\n", tok.decode(one).c_str()); }

    if (!dump_dir.empty()) {
        fs::create_directories(dump_dir);
        for (uint32_t i = 0; i < NC; ++i) {
            char name[64];
            std::snprintf(name, sizeof name, "/call%02u_layer%02u.f32", i, sched[i].layer);
            if (!write_raw(dump_dir + name, res.calls[i].data(), res.calls[i].size())) { std::fprintf(stderr, "write failed\n"); return 1; }
        }
        write_raw(dump_dir + "/logits_last.f32", lg.data(), lg.size());
        json r;
        r["ids"] = ids;
        r["own"] = json::array(); r["used"] = json::array(); r["margin"] = json::array();
        for (uint32_t i = 0; i < NC; ++i) {
            json o = json::array(), u = json::array();
            for (uint32_t t = 0; t < T; ++t) {
                o.push_back(std::vector<int32_t>(res.own_routes[i].begin() + t * 8, res.own_routes[i].begin() + t * 8 + 8));
                u.push_back(std::vector<int32_t>(res.used_routes[i].begin() + t * 8, res.used_routes[i].begin() + t * 8 + 8));
            }
            r["own"].push_back(o); r["used"].push_back(u); r["margin"].push_back(res.route_margin[i]);
        }
        std::ofstream(dump_dir + "/routes.json") << r.dump();
        std::printf("dumped %u calls to %s\n", NC, dump_dir.c_str());
    }
    if (!dump_steps.empty()) {                                          // a single forward: step 0 only
        fs::create_directories(dump_steps);
        write_raw(dump_steps + "/step0.f32", lg.data(), lg.size());
    }
    return 0;
}
