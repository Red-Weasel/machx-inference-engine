// Phase 2a harness (decode-throughput campaign): batched-vs-solo equivalence
// for Qwen35SplitModel::forward_slots on the 2-card split path.
//
// Method: seed TWO identical bank sets (slots 0..N-1 solo, N..2N-1 batched)
// from the same synthetic prefills. Teacher-force the same token stream down
// both: the solo lane steps each slot with forward(T=1) via bank_load/store;
// the batched lane steps all N with forward_slots(). The solo lane runs TWICE
// (independent re-seed) to measure the run-to-run noise floor first — the
// criterion is batched deviation ≤ 2× that floor, plus argmax agreement, plus
// a final bit-compare of bank states between the lanes.
//
// Tokens are synthetic ids (no tokenizer needed — numerical equivalence only).
//
// usage: ie-batched-equiv-test --gguf <27B.gguf> [--gpus 2] [--depth 1024]
//                              [--steps 64] [--slots 4] [--ctx 32768]
#include "ie/allocator.hpp"
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"
#include "ie/qwen35_split.hpp"

#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using ie::Qwen35SplitModel;

static std::vector<int32_t> synth_prompt(uint32_t slot, uint32_t len) {
    std::vector<int32_t> ids(len);
    for (uint32_t t = 0; t < len; ++t)
        ids[t] = int32_t(1000u + (slot * 7919u + t * 2654435761u) % 49000u);
    return ids;
}
static int32_t synth_next(uint32_t slot, uint32_t step) {
    return int32_t(1000u + (slot * 104729u + step * 40503u) % 49000u);
}

int main(int argc, char** argv) {
    std::string gguf;
    uint32_t n_gpus = 2, depth = 1024, steps = 64, n_slots = 4, max_ctx = 32768;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--gguf"  && i + 1 < argc) gguf   = argv[++i];
        else if (a == "--gpus"  && i + 1 < argc) n_gpus = uint32_t(std::atoi(argv[++i]));
        else if (a == "--depth" && i + 1 < argc) depth  = uint32_t(std::atoi(argv[++i]));
        else if (a == "--steps" && i + 1 < argc) steps  = uint32_t(std::atoi(argv[++i]));
        else if (a == "--slots" && i + 1 < argc) n_slots = uint32_t(std::atoi(argv[++i]));
        else if (a == "--ctx"   && i + 1 < argc) max_ctx = uint32_t(std::atoi(argv[++i]));
    }
    if (gguf.empty()) { std::fprintf(stderr, "--gguf required\n"); return 2; }
    const uint32_t slot_ctx = depth + steps + 64;

    ie::GgufReader g;
    if (auto m = g.open(gguf); !m.empty()) { std::fprintf(stderr, "gguf: %s\n", m.c_str()); return 1; }
    ie::Qwen35Config qcfg;
    if (auto m = ie::read_qwen35_config(g, qcfg); !m.empty()) { std::fprintf(stderr, "config: %s\n", m.c_str()); return 1; }
    ie::DeviceFleet fleet;
    if (auto m = fleet.init(n_gpus); !m.empty()) { std::fprintf(stderr, "fleet: %s\n", m.c_str()); return 1; }
    const uint32_t n_tx = qcfg.n_transformer_layers();
    ie::LayerPlan plan = ie::LayerPlan::contiguous(n_tx, n_gpus);
    Qwen35SplitModel model;
    if (auto m = model.load(fleet, plan, g, qcfg, max_ctx, /*int8_kv=*/false); !m.empty()) {
        std::fprintf(stderr, "model: %s\n", m.c_str()); return 1;
    }
    const uint32_t V = qcfg.dense.vocab;
    if (auto m = model.alloc_slot_banks(2 * n_slots, slot_ctx); !m.empty()) {
        std::fprintf(stderr, "banks: %s\n", m.c_str()); return 1;
    }
    std::printf("loaded: %u layers, %u slots x2 lanes, depth %u, steps %u, slot_ctx %u\n",
                n_tx, n_slots, depth, steps, slot_ctx);

    std::vector<sycl::half> logits(V);
    // Seed lane banks: prefill each slot's prompt on live state, store to BOTH.
    auto seed = [&](bool both) -> int {
        for (uint32_t s = 0; s < n_slots; ++s) {
            auto ids = synth_prompt(s, depth);
            if (auto m = model.forward(ids.data(), depth, 0, /*reset_kv=*/true,
                                       logits.data()); !m.empty()) {
                std::fprintf(stderr, "prefill slot %u: %s\n", s, m.c_str()); return 1;
            }
            if (auto m = model.bank_store(s, depth); !m.empty()) {
                std::fprintf(stderr, "store %u: %s\n", s, m.c_str()); return 1;
            }
            if (both)
                if (auto m = model.bank_store(n_slots + s, depth); !m.empty()) {
                    std::fprintf(stderr, "store %u: %s\n", n_slots + s, m.c_str()); return 1;
                }
        }
        return 0;
    };
    // Solo lane: step every slot once per round via bank_load/forward/bank_store.
    auto solo_pass = [&](std::vector<float>& out) -> int {
        out.assign(size_t(steps) * n_slots * V, 0.f);
        for (uint32_t t = 0; t < steps; ++t)
            for (uint32_t s = 0; s < n_slots; ++s) {
                const uint32_t pos = depth + t;
                if (auto m = model.bank_load(s, pos); !m.empty()) { std::fprintf(stderr, "load: %s\n", m.c_str()); return 1; }
                const int32_t tok = synth_next(s, t);
                if (auto m = model.forward(&tok, 1, pos, false, logits.data()); !m.empty()) {
                    std::fprintf(stderr, "solo fwd: %s\n", m.c_str()); return 1;
                }
                if (auto m = model.bank_store(s, pos + 1); !m.empty()) { std::fprintf(stderr, "store: %s\n", m.c_str()); return 1; }
                float* dst = out.data() + (size_t(t) * n_slots + s) * V;
                for (uint32_t v = 0; v < V; ++v) dst[v] = float(logits[v]);
            }
        return 0;
    };

    std::vector<float> soloA, soloB, batched;
    // Order matters: lanes A and B get the SAME seed, then solo runs on A and
    // batched on B, and their states are compared while still twins. The noise
    // floor (re-seed + second solo pass) runs LAST — a fresh prefill breaks
    // twin-ness (prefill is where run-to-run nondeterminism enters).
    if (seed(true)) return 1;
    if (solo_pass(soloA)) return 1;

    // Batched lane (timed: ms/step and implied aggregate tok/s at this depth).
    batched.assign(size_t(steps) * n_slots * V, 0.f);
    std::vector<sycl::half> blog(size_t(n_slots) * V);
    std::vector<int32_t>  toks(n_slots);
    std::vector<uint32_t> pos(n_slots), slots(n_slots);
    const auto bt0 = std::chrono::steady_clock::now();
    for (uint32_t t = 0; t < steps; ++t) {
        for (uint32_t s = 0; s < n_slots; ++s) {
            toks[s] = synth_next(s, t);
            pos[s] = depth + t;
            slots[s] = n_slots + s;
        }
        if (auto m = model.forward_slots(n_slots, toks.data(), pos.data(),
                                         slots.data(), blog.data()); !m.empty()) {
            std::fprintf(stderr, "forward_slots: %s\n", m.c_str()); return 1;
        }
        for (uint32_t s = 0; s < n_slots; ++s) {
            float* dst = batched.data() + (size_t(t) * n_slots + s) * V;
            for (uint32_t v = 0; v < V; ++v) dst[v] = float(blog[size_t(s) * V + v]);
        }
    }
    {
        const double bms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - bt0).count();
        std::printf("batched timing: %.1f ms/step (%u slots) → aggregate %.1f tok/s "
                    "at depth ~%u (incl. host logits bounce %u x %u fp16/step)\n",
                    bms / steps, n_slots, 1000.0 * n_slots * steps / bms,
                    depth, n_slots, V);
    }

    // Noise floor AFTER the state compare below reads the twin lanes — so
    // compute the state verdict first, then re-seed for the floor.
    auto maxdiff = [&](const std::vector<float>& a, const std::vector<float>& b) {
        float m = 0.f;
        for (size_t i = 0; i < a.size(); ++i) m = std::fmax(m, std::fabs(a[i] - b[i]));
        return m;
    };
    auto argmax_mismatch = [&](const std::vector<float>& a, const std::vector<float>& b) {
        uint32_t bad = 0;
        for (uint32_t t = 0; t < steps; ++t)
            for (uint32_t s = 0; s < n_slots; ++s) {
                const float* pa = a.data() + (size_t(t) * n_slots + s) * V;
                const float* pb = b.data() + (size_t(t) * n_slots + s) * V;
                uint32_t ia = 0, ib = 0;
                for (uint32_t v = 1; v < V; ++v) {
                    if (pa[v] > pa[ia]) ia = v;
                    if (pb[v] > pb[ib]) ib = v;
                }
                if (ia != ib) ++bad;
            }
        return bad;
    };
    // Final-state bit-compare FIRST, while lanes are still twins: solo lane
    // bank s vs batched lane bank N+s — same seed, same forced stream.
    bool state_equal = true;
    {
        const uint32_t fin = depth + steps;
        for (uint32_t s = 0; s < n_slots && state_equal; ++s)
            for (uint32_t d = 0; d < n_gpus && state_equal; ++d) {
                sycl::queue& q = fleet.dev(d).queue();
                if (model.dev_has_kv(d)) {
                    ie::KvCache& a = model.kv_cache(d);   // layout source only
                    (void)a;
                }
                // Compare via host downloads of the banks' K/V prefix + DN.
                auto cmp_kv = [&](ie::KvCache& x, ie::KvCache& y) {
                    const auto& kc = x.config();
                    const uint64_t slice = uint64_t(fin) * kc.head_dim * sizeof(sycl::half);
                    const uint64_t stride = uint64_t(kc.max_ctx) * kc.head_dim;
                    const uint64_t nsl = uint64_t(kc.n_layers_full) * kc.n_kv_heads;
                    std::vector<uint8_t> ha(slice), hb(slice);
                    for (uint64_t r = 0; r < nsl; ++r) {
                        q.memcpy(ha.data(), x.k_ptr() + r * stride, slice).wait();
                        q.memcpy(hb.data(), y.k_ptr() + r * stride, slice).wait();
                        if (ha != hb) return false;
                        q.memcpy(ha.data(), x.v_ptr() + r * stride, slice).wait();
                        q.memcpy(hb.data(), y.v_ptr() + r * stride, slice).wait();
                        if (ha != hb) return false;
                    }
                    return true;
                };
                auto cmp_dn = [&](ie::DeltaNetState& x, ie::DeltaNetState& y) {
                    const uint64_t sb = x.state_elems_per_layer() *
                                        x.config().n_layers_linear * sizeof(float);
                    const uint64_t cb = x.conv_elems_per_layer() *
                                        x.config().n_layers_linear * sizeof(sycl::half);
                    std::vector<uint8_t> ha(sb), hb(sb);
                    q.memcpy(ha.data(), x.state_ptr(), sb).wait();
                    q.memcpy(hb.data(), y.state_ptr(), sb).wait();
                    if (ha != hb) return false;
                    ha.resize(cb); hb.resize(cb);
                    q.memcpy(ha.data(), x.conv_state_ptr(), cb).wait();
                    q.memcpy(hb.data(), y.conv_state_ptr(), cb).wait();
                    return ha == hb;
                };
                ie::KvCache&       ka = model.bank_kv(s, d);
                ie::KvCache&       kb = model.bank_kv(n_slots + s, d);
                ie::DeltaNetState& da = model.bank_dn(s, d);
                ie::DeltaNetState& db = model.bank_dn(n_slots + s, d);
                if (ka.ready() && !cmp_kv(ka, kb)) state_equal = false;
                if (state_equal && da.ready() && !cmp_dn(da, db)) state_equal = false;
            }
    }
    std::printf("final bank state bit-compare: %s\n", state_equal ? "EQUAL" : "DIFFERS");

    // Noise floor last: a fresh re-seed of the solo lane + second solo pass.
    if (seed(false)) return 1;
    if (solo_pass(soloB)) return 1;
    const float noise = maxdiff(soloA, soloB);
    const float dev   = maxdiff(soloA, batched);
    const uint32_t am_noise = argmax_mismatch(soloA, soloB);
    const uint32_t am_dev   = argmax_mismatch(soloA, batched);
    std::printf("noise floor (solo vs re-seeded solo): max|Δ| %.6f, argmax mismatches %u/%u\n",
                noise, am_noise, steps * n_slots);
    std::printf("batched vs solo (twin lanes):         max|Δ| %.6f, argmax mismatches %u/%u\n",
                dev, am_dev, steps * n_slots);

    const bool pass = (dev <= 2.0f * noise || dev == 0.f) &&
                      am_dev <= am_noise && state_equal;
    std::printf("%s: batched deviation %s noise floor, argmax %u vs %u, state %s\n",
                pass ? "PASS" : "FAIL", dev <= 2.0f * noise ? "within 2x" : "EXCEEDS 2x",
                am_dev, am_noise, state_equal ? "bit-equal" : "diverged");
    return pass ? 0 : 1;
}
