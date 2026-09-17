// tools/ds41_probe.cpp — DeepSeek-V4.1-Flash Phase 2 probe (host-only, no GPU).
//
// Answers exactly one question: can this engine read the shipped checkpoint's bytes and get
// the same numbers the reference implementation gets? It opens the 48-shard safetensors
// checkpoint through `ie::SafetensorsModel`, audits discovery/resolution/size-math, then
// dequantises one dense FP8 tensor and decodes one routed FP4 expert and writes both as raw
// fp32 for `tools/ds41_probe_ref.py` to diff against torch's own float8_e4m3fn /
// float8_e8m0fnu decode of the SAME bytes.
//
// Criteria: docs/deepseek41/01_PHASE2_CRITERIA_2026-09-12.md
#include "ie/safetensors.hpp"
#include "ie/dtype.hpp"
#include "ie/fp8.hpp"

#include "../third_party/nlohmann/json.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using json = nlohmann::json;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what, const std::string& detail = "") {
    const std::string line = std::string(ok ? "[ ok ] " : "[FAIL] ") + what +
                             (detail.empty() ? "" : "  (" + detail + ")");
    std::printf("%s\n", line.c_str());
    if (!ok) ++g_fail;
}

// The engine's MXFP4 nibble decode, transcribed from src/ops/deepseek4_experts.cpp:112.
// The table yields 2x the e2m1 magnitude, so the 0.5f here is the engine's own halving.
float mxfp4_nibble(uint8_t nb) {
    const int mag = int((0xC8643210u >> ((nb & 7u) * 4u)) & 0xFu);
    const float v = float(mag) * 0.5f;
    return (nb & 8u) ? -v : v;
}

void dump(const std::string& path, const std::vector<float>& v) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * 4));
}

const ie::SafeTensorInfo* need(const ie::SafetensorsModel& m, const std::string& n) {
    const auto* t = m.find(n);
    if (!t) { std::printf("[FAIL] missing tensor %s\n", n.c_str()); ++g_fail; }
    return t;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : std::string(std::getenv("HOME")) + "/models/DeepSeek-V4.1-Flash";
    const std::string out = argc > 2 ? argv[2] : "/tmp/ds41_probe";

    // ---- C1: discovery -----------------------------------------------------------------
    ie::SafetensorsModel model;
    if (const auto e = model.open(dir); !e.empty()) {
        std::fprintf(stderr, "open(%s): %s\n", dir.c_str(), e.c_str());
        return 1;
    }
    const auto all = model.all();
    std::printf("=== C1 discovery ===\n");

    json idx;
    {
        std::ifstream f(dir + "/model.safetensors.index.json", std::ios::binary);
        if (!f) { std::fprintf(stderr, "no index.json\n"); return 1; }
        idx = json::parse(std::string((std::istreambuf_iterator<char>(f)),
                                       std::istreambuf_iterator<char>()));
    }
    const auto& wm = idx["weight_map"];
    std::unordered_set<std::string> shard_names;
    for (auto it = wm.begin(); it != wm.end(); ++it)
        shard_names.insert(it.value().get<std::string>());

    check(model.shard_count() == shard_names.size(), "shard count == index.json shard count",
          std::to_string(model.shard_count()) + " vs " + std::to_string(shard_names.size()));
    check(all.size() == wm.size(), "tensor count == weight_map size",
          std::to_string(all.size()) + " vs " + std::to_string(wm.size()));

    // ---- C2: resolution is total and correct -------------------------------------------
    // Precise guard on the linear-scan -> hash-index rewrite in SafetensorsReader::find:
    //  (a) within a shard, find(name) must return the SAME object, by pointer, not merely a
    //      tensor with that name;  (b) no name may appear in two shards, or first-wins at the
    //      model level is ambiguous;  (c) every weight_map name resolves.
    std::printf("\n=== C2 resolution ===\n");
    {
        size_t ident_bad = 0;
        std::unordered_map<std::string, int> seen;
        seen.reserve(all.size() * 2);
        for (const auto* t : all) ++seen[t->name];
        size_t dup = 0;
        for (const auto& kv : seen) if (kv.second != 1) ++dup;
        check(dup == 0, "no tensor name appears in two shards", std::to_string(dup) + " duplicated");

        // Pointer identity, exhaustively, through the model's own find().
        for (const auto* t : all)
            if (model.find(t->name) != t) ++ident_bad;
        check(ident_bad == 0, "find(name) returns the identical object for all tensors",
              std::to_string(ident_bad) + " mismatched of " + std::to_string(all.size()));

        size_t missing = 0;
        for (auto it = wm.begin(); it != wm.end(); ++it)
            if (!model.find(it.key())) ++missing;
        check(missing == 0, "every weight_map name resolves", std::to_string(missing) + " missing");
    }

    // ---- C3: size math ------------------------------------------------------------------
    std::printf("\n=== C3 dtype census + bytes_for ===\n");
    {
        std::map<std::string, std::pair<size_t, uint64_t>> census;  // dtype_str -> (count, bytes)
        size_t bad = 0; std::string first_bad;
        for (const auto* t : all) {
            auto& c = census[t->dtype_str];
            c.first += 1; c.second += t->nbytes;
            if (t->dtype == ie::DType::kCount) continue;
            const size_t want = ie::bytes_for(t->dtype, size_t(t->numel()));
            if (want != t->nbytes) {
                if (!bad) first_bad = t->name + " " + t->dtype_str + " want " +
                                      std::to_string(want) + " got " + std::to_string(t->nbytes);
                ++bad;
            }
        }
        for (const auto& kv : census)
            std::printf("       %-10s %8zu tensors  %10.2f GiB   engine dtype %s\n",
                        kv.first.c_str(), kv.second.first, double(kv.second.second) / (1<<30),
                        std::string(ie::type_name(ie::SafetensorsReader::dtype_from_string(kv.first))).c_str());
        check(bad == 0, "bytes_for(dtype, numel) == shipped nbytes for every mapped tensor",
              bad ? first_bad : "all " + std::to_string(all.size()) + " checked");

        // FP4 expert planes ship as U8/I8 at [N, K/2]; retyped by role, the size math must
        // still land on the shipped byte count.
        const auto* w1 = need(model, "layers.0.ffn.experts.0.w1.weight");
        if (w1 && w1->shape.size() == 2) {
            const size_t N = size_t(w1->shape[0]), Kh = size_t(w1->shape[1]);
            check(ie::bytes_for(ie::DType::kFP4_E2M1, N * Kh * 2) == w1->nbytes,
                  "bytes_for(kFP4_E2M1, N*K) == shipped nbytes for a retyped expert plane",
                  "N=" + std::to_string(N) + " K=" + std::to_string(Kh * 2));
        }
    }

    // ---- C4: dense dequant ---------------------------------------------------------------
    std::printf("\n=== C4 dense FP8 dequant (layers.0.attn.wq_b) ===\n");
    {
        const auto* w = need(model, "layers.0.attn.wq_b.weight");
        const auto* s = need(model, "layers.0.attn.wq_b.scale");
        if (w && s && w->shape.size() == 2 && s->shape.size() == 2) {
            const size_t N = size_t(w->shape[0]), K = size_t(w->shape[1]);
            const size_t SN = size_t(s->shape[0]), SK = size_t(s->shape[1]);
            const size_t bn = N / SN, bk = K / SK;
            check(N % SN == 0 && K % SK == 0, "scale grid tiles the weight",
                  std::to_string(bn) + "x" + std::to_string(bk));
            check(w->dtype == ie::DType::kFP8_E4M3 && s->dtype == ie::DType::kE8M0,
                  "dtypes map to kFP8_E4M3 / kE8M0",
                  w->dtype_str + " / " + s->dtype_str);
            std::vector<float> deq(N * K);
            for (size_t n = 0; n < N; ++n)
                for (size_t k = 0; k < K; ++k)
                    deq[n * K + k] = ie::e4m3_to_f32(w->data[n * K + k]) *
                                     ie::e8m0_to_f32(s->data[(n / bn) * SK + (k / bk)]);
            dump(out + "/dense.f32", deq);
            std::printf("       wrote %s/dense.f32  [%zu, %zu]\n", out.c_str(), N, K);
        }
    }

    // ---- C5: expert bind -----------------------------------------------------------------
    std::printf("\n=== C5 routed FP4 expert decode (layers.0.ffn.experts.0.w1) ===\n");
    {
        const auto* w = need(model, "layers.0.ffn.experts.0.w1.weight");
        const auto* s = need(model, "layers.0.ffn.experts.0.w1.scale");
        if (w && s && w->shape.size() == 2 && s->shape.size() == 2) {
            const size_t N = size_t(w->shape[0]), Kh = size_t(w->shape[1]), K = Kh * 2;
            check(size_t(s->shape[0]) == N && size_t(s->shape[1]) == K / 32,
                  "scale plane is [N, K/32]",
                  "N=" + std::to_string(N) + " K=" + std::to_string(K) +
                  " scale=[" + std::to_string(s->shape[0]) + "," + std::to_string(s->shape[1]) + "]");
            std::vector<float> deq(N * K);
            for (size_t n = 0; n < N; ++n)
                for (size_t b = 0; b < Kh; ++b) {
                    const uint8_t byte = w->data[n * Kh + b];
                    const size_t k = b * 2;
                    deq[n * K + k]     = mxfp4_nibble(byte & 0x0Fu) *
                                         ie::e8m0_to_f32(s->data[n * (K / 32) + k / 32]);
                    deq[n * K + k + 1] = mxfp4_nibble(byte >> 4) *
                                         ie::e8m0_to_f32(s->data[n * (K / 32) + (k + 1) / 32]);
                }
            dump(out + "/expert.f32", deq);
            std::printf("       wrote %s/expert.f32  [%zu, %zu]\n", out.c_str(), N, K);

            // Nibble ORDER, not just magnitude: find a byte whose two nibbles decode to
            // different magnitudes and report it so the reference can confirm which element
            // got which. An order flip changes this pair and nothing else detects it.
            for (size_t n = 0; n < N; ++n) {
                bool done = false;
                for (size_t b = 0; b < Kh && !done; ++b) {
                    const uint8_t by = w->data[n * Kh + b];
                    const float lo = mxfp4_nibble(by & 0x0Fu), hi = mxfp4_nibble(by >> 4);
                    if (lo != hi && lo != 0.0f && hi != 0.0f) {
                        std::printf("       order probe: byte[%zu,%zu]=0x%02X -> "
                                    "elem[%zu,%zu]=%g  elem[%zu,%zu]=%g\n",
                                    n, b, by, n, b * 2, double(deq[n * K + b * 2]),
                                    n, b * 2 + 1, double(deq[n * K + b * 2 + 1]));
                        done = true;
                    }
                }
                if (done) break;
            }
        }
    }

    // Freshness stamp, written LAST. ie-ds41-probe and ds41_probe_ref.py are separate
    // processes with no handshake, so without this a stale dump left in a reused directory
    // would be compared silently and "pass". The reference script refuses a stale stamp.
    {
        const auto now = std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::system_clock::now().time_since_epoch()).count();
        std::ofstream f(out + "/stamp", std::ios::trunc);
        f << now << "\n";
    }

    std::printf("\n%s\n", g_fail ? ("PROBE: " + std::to_string(g_fail) + " FAILURE(S)").c_str()
                                 : "PROBE (C1-C3, C5 shape): PASS — run ds41_probe_ref.py for C4/C5 numerics");
    return g_fail ? 1 : 0;
}
