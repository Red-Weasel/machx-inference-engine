// tools/deepseek4/ds4_mini_gguf.hpp — the miniature but STRUCTURALLY REAL
// deepseek4 GGUF fixture.
//
// LIFTED VERBATIM out of tests/unit/deepseek4_forward_test.cpp's `namespace
// mini`, which is still its only behavioural owner: §6/§7 of that gate are this
// fixture's output, and the comment below about `iq3_scale` is why the defaults
// here are frozen.  It moved into a header for ONE reason — tools/ie_ds4_bench.cpp
// needs the SAME fixture, and a second copy of a 230-line generator is a second
// thing that can drift.  Every function is `inline`; nothing else changed.
//
// WHY.  The 128.20 GB production file lives on a USB-2 spinning disk that this
// session measured at 1.4-12.4 MB/s, so an end-to-end run against it is hours of
// I/O, not minutes.  Without something else to run, the assembled forward pass
// (and now the benchmark harness over it) would ship having never executed.
// This writes a real GGUF that the engine's own GgufReader parses and
// DeepSeek4Model binds with the SAME 1:1 tensor contract as the production file —
// every tensor name, every dtype (Q8_0 / Q6_K / BF16 / I32 / F32 / IQ3_XXS /
// MXFP4), all three attention layer types, both routers, and one mixed-precision
// all-MXFP4 expert layer standing in for blk.26.  Only the DIMENSIONS shrink.
//
// WHAT IT PROVES AND WHAT IT DOES NOT.  It proves the code paths execute on the
// GPU, allocate and stream correctly, and produce finite, non-degenerate logits.
// The weights are pseudo-random, so it proves NOTHING about accuracy against
// DeepSeek's trained weights — and, for the benchmark harness, NOTHING about the
// real model's throughput: 6 layers of hidden 256 with 8 experts is a different
// machine from 43 layers of hidden 4096 with 256.  Numbers measured here are
// evidence that the METRICS ARE COMPUTED, never evidence of a performance level.
#pragma once

#include "ie/gguf_writer.hpp"
#include "ie/model_config.hpp"
#include "ie/quant_blocks.hpp"

#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace ds4mini {

struct Cfg {
    uint32_t L = 6, H = 256, NH = 4, HD = 64, VOCAB = 512;
    uint32_t QR = 256, OLR = 64, OG = 2, SW = 16;
    uint32_t IH = 4, IHD = 64, ITOPK = 8;
    uint32_t E = 8, EU = 2, EFF = 256;
    uint32_t HC = 4, SINK = 20, HASH = 3, RD = 32;
    std::vector<int32_t> ratios{0, 0, 4, 128, 4, 128};
    uint32_t mxfp4_layer = 4;      // stands in for blk.26's UD upcast
    // Routed-expert IQ3_XXS super-scale.  THE DEFAULT IS THE SHIPPED §6 FIXTURE
    // AND MUST NOT CHANGE — §6's logits and greedy ids are its output.
    //
    // It is a knob because §7 needs the same model at a second conditioning.  At
    // 0.02 this fixture's clamped SwiGLU sits permanently AGAINST its clamp, so
    // the routed-expert block is a near-discontinuous function of its input and
    // a last-bit difference anywhere upstream lands on the far side of a clamp.
    // That makes §6 a fine exerciser of the code paths and a useless instrument
    // for comparing two numerically-equivalent-but-not-bit-identical paths.
    // deepseek4_residency_test's own fixture is this one with the scale at
    // 5e-4 for exactly that reason (see its §15 comment).
    float iq3_scale = 0.02f;
};

struct Buf { std::vector<uint8_t> b; };

// Deterministic small floats — real weights are small, and random fp16/bf16
// BIT patterns would be Inf/NaN, which would make "finite logits" meaningless.
struct Rng {
    std::mt19937 g;
    explicit Rng(uint64_t s) : g(uint32_t(s)) {}
    float sn(float sd) { std::normal_distribution<float> d(0.f, sd); return d(g); }
    uint32_t u(uint32_t n) { return g() % n; }
};

inline std::vector<uint8_t> f32_buf(Rng& r, size_t n, float sd, float bias = 0.f) {
    std::vector<uint8_t> b(n * 4);
    auto* p = reinterpret_cast<float*>(b.data());
    for (size_t i = 0; i < n; ++i) p[i] = r.sn(sd) + bias;
    return b;
}
inline std::vector<uint8_t> bf16_buf(Rng& r, size_t n, float sd) {
    std::vector<uint8_t> b(n * 2);
    auto* p = reinterpret_cast<uint16_t*>(b.data());
    for (size_t i = 0; i < n; ++i) {
        const float f = r.sn(sd);
        uint32_t bits; std::memcpy(&bits, &f, 4);
        p[i] = uint16_t(bits >> 16);
    }
    return b;
}
inline std::vector<uint8_t> i32_buf(Rng& r, size_t n, uint32_t mod) {
    std::vector<uint8_t> b(n * 4);
    auto* p = reinterpret_cast<int32_t*>(b.data());
    for (size_t i = 0; i < n; ++i) p[i] = int32_t(r.u(mod));
    return b;
}
// Valid Q8_0: a sane fp16 scale, int8 quants.
inline std::vector<uint8_t> q8_buf(Rng& r, size_t n) {
    const size_t nb = n / 32;
    std::vector<uint8_t> b(nb * sizeof(ie::block_q8_0));
    auto* blk = reinterpret_cast<ie::block_q8_0*>(b.data());
    for (size_t i = 0; i < nb; ++i) {
        blk[i].d = ie::fp32_to_fp16(0.02f);
        for (int j = 0; j < 32; ++j) blk[i].qs[j] = int8_t(int(r.u(65)) - 32);
    }
    return b;
}
// Valid Q6_K: bounded super-scale and per-16 scales.
inline std::vector<uint8_t> q6k_buf(Rng& r, size_t n) {
    const size_t nb = n / 256;
    std::vector<uint8_t> b(nb * sizeof(ie::block_q6_K));
    auto* blk = reinterpret_cast<ie::block_q6_K*>(b.data());
    for (size_t i = 0; i < nb; ++i) {
        blk[i].d = ie::fp32_to_fp16(0.004f);
        for (int j = 0; j < 128; ++j) blk[i].ql[j] = uint8_t(r.u(256));
        for (int j = 0; j < 64; ++j)  blk[i].qh[j] = uint8_t(r.u(256));
        for (int j = 0; j < 16; ++j)  blk[i].scales[j] = int8_t(int(r.u(17)) - 8);
    }
    return b;
}
// Valid IQ3_XXS: bounded fp16 super-scale; the 96 payload bytes are free
// (grid indices index a 256-entry table, the scale nibble is 4 bits).
inline std::vector<uint8_t> iq3_buf(Rng& r, size_t n, float scale) {
    const size_t nb = n / 256;
    std::vector<uint8_t> b(nb * sizeof(ie::block_iq3_xxs));
    auto* blk = reinterpret_cast<ie::block_iq3_xxs*>(b.data());
    for (size_t i = 0; i < nb; ++i) {
        blk[i].d = ie::fp32_to_fp16(scale);
        for (int j = 0; j < 96; ++j) blk[i].qs[j] = uint8_t(r.u(256));
    }
    return b;
}
// Valid MXFP4: E8M0 exponent kept near 127, otherwise 2^(e-127) overflows.
inline std::vector<uint8_t> mxfp4_buf(Rng& r, size_t n) {
    const size_t nb = n / 32;
    std::vector<uint8_t> b(nb * sizeof(ie::block_mxfp4));
    auto* blk = reinterpret_cast<ie::block_mxfp4*>(b.data());
    for (size_t i = 0; i < nb; ++i) {
        blk[i].e = uint8_t(124 + r.u(3));
        for (int j = 0; j < 16; ++j) blk[i].qs[j] = uint8_t(r.u(256));
    }
    return b;
}

inline std::string write_gguf(const Cfg& c, const std::string& path,
                              std::vector<std::vector<uint8_t>>& keep) {
    using namespace ie;
    Rng r(20260801);
    GgufWriter w;
    w.kv_string("general.architecture", "deepseek4");
    w.kv_u32("deepseek4.block_count", c.L);
    w.kv_u32("deepseek4.embedding_length", c.H);
    w.kv_u32("deepseek4.attention.head_count", c.NH);
    w.kv_u32("deepseek4.attention.head_count_kv", 1);
    w.kv_u32("deepseek4.attention.key_length", c.HD);
    w.kv_u32("deepseek4.attention.value_length", c.HD);
    w.kv_u32("deepseek4.context_length", 4096);
    w.kv_f32("deepseek4.attention.layer_norm_rms_epsilon", 1e-6f);
    w.kv_u32("deepseek4.attention.q_lora_rank", c.QR);
    w.kv_u32("deepseek4.attention.output_lora_rank", c.OLR);
    w.kv_u32("deepseek4.attention.output_group_count", c.OG);
    w.kv_u32("deepseek4.attention.sliding_window", c.SW);
    w.kv_u32("deepseek4.attention.indexer.head_count", c.IH);
    w.kv_u32("deepseek4.attention.indexer.key_length", c.IHD);
    w.kv_u32("deepseek4.attention.indexer.top_k", c.ITOPK);
    w.kv_i32_array("deepseek4.attention.compress_ratios", c.ratios);
    w.kv_f32("deepseek4.attention.compress_rope_freq_base", 160000.f);
    w.kv_u32("deepseek4.rope.dimension_count", c.RD);
    w.kv_f32("deepseek4.rope.freq_base", 10000.f);
    w.kv_f32("deepseek4.rope.scaling.factor", 16.f);
    w.kv_string("deepseek4.rope.scaling.type", "yarn");
    w.kv_u32("deepseek4.rope.scaling.original_context_length", 65536);
    w.kv_f32("deepseek4.rope.scaling.yarn_beta_fast", 32.f);
    w.kv_f32("deepseek4.rope.scaling.yarn_beta_slow", 1.f);
    w.kv_u32("deepseek4.expert_count", c.E);
    w.kv_u32("deepseek4.expert_used_count", c.EU);
    w.kv_u32("deepseek4.expert_feed_forward_length", c.EFF);
    w.kv_u32("deepseek4.expert_shared_count", 1);
    w.kv_u32("deepseek4.expert_gating_func", 4);
    w.kv_f32("deepseek4.expert_weights_scale", 1.5f);
    w.kv_bool("deepseek4.expert_weights_norm", true);
    w.kv_f32_array("deepseek4.swiglu_clamp_exp", std::vector<float>(c.L, 10.f));
    w.kv_f32_array("deepseek4.swiglu_clamp_shexp", std::vector<float>(c.L, 10.f));
    w.kv_u32("deepseek4.hyper_connection.count", c.HC);
    w.kv_u32("deepseek4.hyper_connection.sinkhorn_iterations", c.SINK);
    w.kv_f32("deepseek4.hyper_connection.epsilon", 1e-6f);
    w.kv_u32("deepseek4.hash_layer_count", c.HASH);
    std::vector<std::string> toks(c.VOCAB);
    for (uint32_t i = 0; i < c.VOCAB; ++i) toks[i] = "t" + std::to_string(i);
    w.kv_string_array("tokenizer.ggml.tokens", toks);
    w.kv_string("tokenizer.ggml.model", "gpt2");

    auto add = [&](const std::string& nm, DType dt, std::vector<uint64_t> shape,
                   std::vector<uint8_t>&& data) {
        keep.push_back(std::move(data));
        w.tensor(nm, dt, shape, keep.back().data(), keep.back().size());
    };
    const uint32_t MIX = (2 + c.HC) * c.HC, HCH = c.HC * c.H, ORK = c.OLR * c.OG;

    add("token_embd.weight", DType::kQ8_0, {c.H, c.VOCAB}, q8_buf(r, size_t(c.H) * c.VOCAB));
    add("output.weight", DType::kQ6_K, {c.H, c.VOCAB}, q6k_buf(r, size_t(c.H) * c.VOCAB));
    add("output_norm.weight", DType::kF32, {c.H}, f32_buf(r, c.H, 0.02f, 1.f));
    add("output_hc_fn.weight", DType::kF32, {HCH, c.HC}, f32_buf(r, size_t(HCH) * c.HC, 0.02f));
    add("output_hc_base.weight", DType::kF32, {c.HC}, f32_buf(r, c.HC, 0.1f));
    add("output_hc_scale.weight", DType::kF32, {1}, f32_buf(r, 1, 0.f, 1.f));

    for (uint32_t L = 0; L < c.L; ++L) {
        const std::string b = "blk." + std::to_string(L) + ".";
        const int32_t ratio = c.ratios[L];
        const uint32_t CD = ratio == 4 ? 2 * c.HD : c.HD;
        add(b + "attn_norm.weight", DType::kF32, {c.H}, f32_buf(r, c.H, 0.02f, 1.f));
        add(b + "attn_q_a.weight", DType::kQ8_0, {c.H, c.QR}, q8_buf(r, size_t(c.H) * c.QR));
        add(b + "attn_q_a_norm.weight", DType::kF32, {c.QR}, f32_buf(r, c.QR, 0.02f, 1.f));
        add(b + "attn_q_b.weight", DType::kQ8_0, {c.QR, c.NH * c.HD},
            q8_buf(r, size_t(c.QR) * c.NH * c.HD));
        add(b + "attn_kv.weight", DType::kQ8_0, {c.H, c.HD}, q8_buf(r, size_t(c.H) * c.HD));
        add(b + "attn_kv_a_norm.weight", DType::kF32, {c.HD}, f32_buf(r, c.HD, 0.02f, 1.f));
        add(b + "attn_sinks.weight", DType::kF32, {c.NH}, f32_buf(r, c.NH, 0.5f));
        add(b + "attn_output_a.weight", DType::kQ8_0, {c.H, ORK}, q8_buf(r, size_t(c.H) * ORK));
        add(b + "attn_output_b.weight", DType::kQ8_0, {ORK, c.H}, q8_buf(r, size_t(ORK) * c.H));
        if (ratio) {
            add(b + "attn_compressor_kv.weight", DType::kQ8_0, {c.H, CD}, q8_buf(r, size_t(c.H) * CD));
            add(b + "attn_compressor_gate.weight", DType::kQ8_0, {c.H, CD}, q8_buf(r, size_t(c.H) * CD));
            add(b + "attn_compressor_ape.weight", DType::kF32, {CD, uint64_t(ratio)},
                f32_buf(r, size_t(CD) * ratio, 0.1f));
            add(b + "attn_compressor_norm.weight", DType::kF32, {c.HD}, f32_buf(r, c.HD, 0.02f, 1.f));
        }
        if (ratio == 4) {
            add(b + "indexer.attn_q_b.weight", DType::kQ8_0, {c.QR, c.IH * c.IHD},
                q8_buf(r, size_t(c.QR) * c.IH * c.IHD));
            add(b + "indexer.proj.weight", DType::kF32, {c.H, c.IH}, f32_buf(r, size_t(c.H) * c.IH, 0.05f));
            add(b + "indexer_compressor_kv.weight", DType::kQ8_0, {c.H, 2 * c.IHD},
                q8_buf(r, size_t(c.H) * 2 * c.IHD));
            add(b + "indexer_compressor_gate.weight", DType::kQ8_0, {c.H, 2 * c.IHD},
                q8_buf(r, size_t(c.H) * 2 * c.IHD));
            add(b + "indexer_compressor_ape.weight", DType::kF32, {2 * c.IHD, uint64_t(ratio)},
                f32_buf(r, size_t(2 * c.IHD) * ratio, 0.1f));
            add(b + "indexer_compressor_norm.weight", DType::kF32, {c.IHD}, f32_buf(r, c.IHD, 0.02f, 1.f));
        }
        add(b + "ffn_norm.weight", DType::kF32, {c.H}, f32_buf(r, c.H, 0.02f, 1.f));
        add(b + "ffn_gate_inp.weight", DType::kBF16, {c.H, c.E}, bf16_buf(r, size_t(c.H) * c.E, 0.05f));
        // The blk.26 stand-in: this ONE layer's gate/up are MXFP4 while every
        // other layer's are IQ3_XXS, so the per-tensor dtype dispatch is
        // exercised end to end (bind -> slot layout -> pack -> stream -> GEMV).
        const bool mx = (L == c.mxfp4_layer);
        const size_t ge = size_t(c.H) * c.EFF * c.E;
        add(b + "ffn_gate_exps.weight", mx ? DType::kMXFP4 : DType::kIQ3_XXS, {c.H, c.EFF, c.E},
            mx ? mxfp4_buf(r, ge) : iq3_buf(r, ge, c.iq3_scale));
        add(b + "ffn_up_exps.weight", mx ? DType::kMXFP4 : DType::kIQ3_XXS, {c.H, c.EFF, c.E},
            mx ? mxfp4_buf(r, ge) : iq3_buf(r, ge, c.iq3_scale));
        add(b + "ffn_down_exps.weight", DType::kMXFP4, {c.EFF, c.H, c.E}, mxfp4_buf(r, ge));
        add(b + "ffn_gate_shexp.weight", DType::kQ8_0, {c.H, c.EFF}, q8_buf(r, size_t(c.H) * c.EFF));
        add(b + "ffn_up_shexp.weight", DType::kQ8_0, {c.H, c.EFF}, q8_buf(r, size_t(c.H) * c.EFF));
        add(b + "ffn_down_shexp.weight", DType::kQ8_0, {c.EFF, c.H}, q8_buf(r, size_t(c.EFF) * c.H));
        if (L < c.HASH)
            add(b + "ffn_gate_tid2eid.weight", DType::kI32, {c.EU, c.VOCAB},
                i32_buf(r, size_t(c.EU) * c.VOCAB, c.E));
        else
            add(b + "exp_probs_b.bias", DType::kF32, {c.E}, f32_buf(r, c.E, 0.05f));
        for (const char* site : {"hc_attn", "hc_ffn"}) {
            add(b + site + "_fn.weight", DType::kF32, {HCH, MIX},
                f32_buf(r, size_t(HCH) * MIX, 0.02f));
            add(b + site + "_base.weight", DType::kF32, {MIX}, f32_buf(r, MIX, 0.1f));
            add(b + site + "_scale.weight", DType::kF32, {3}, f32_buf(r, 3, 0.f, 1.f));
        }
    }
    return w.write(path);
}

}  // namespace ds4mini
