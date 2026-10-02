// src/model/q35_xmx_decode.hpp -- P4 B23: the XMX flash-decode at head_dim 256 for the qwen35 (27B) and qwen35moe
// (35B-A3B) splits. ON by default (owner decision 2026-09-29, after the B23 gate PASS: +38 % 35B / +6-8 % 27B at 32K,
// PPL neutral); IE_Q35_XMX_DECODE=0 turns it off (the pre-B23 bytes). IE_Q35_XMX_DECODE_MIN (default 4096) is the context
// length (start_pos + 1) from which the XMX kernel runs -- below it the existing kernel (vec on the 27B, SLM-tiled on the
// 35B) is faster (B22 bench: XMX has a 50-65 us floor, wins from ~4K: 27B shape 33K 858 vs 1054 us, 35B 667 vs 1240).
// The XMX kernel's numerics differ from both (fp16 P weights, another summation order; max |y - y_vec| ~ 1 fp16 ulp):
// greedy replies past the threshold can flip at near-ties. Its append pass zeroes the 15 cache rows after the token, so
// it also needs start_pos + 16 <= max_ctx; the last 15 positions of a cache stay on the existing kernel (full_attention_fa2_decode_xmx itself would fall
// to the tiled kernel there, a third numerics on the 27B).
#pragma once

#include <cstdint>
#include <cstdlib>

namespace ie::q35 {

inline uint32_t xmx_decode_min() {
    static const uint32_t v = [] {
        const char* e = std::getenv("IE_Q35_XMX_DECODE");
        if (e && *e && std::atoi(e) == 0) return 0u;
        const char* m = std::getenv("IE_Q35_XMX_DECODE_MIN");
        const int mi = m ? std::atoi(m) : 4096;
        return mi > 0 ? uint32_t(mi) : 4096u;
    }();
    return v;
}

// true = run full_attention_fa2_decode_xmx for the token at start_pos of a cache of max_ctx rows.
inline bool xmx_decode(uint32_t start_pos, uint32_t max_ctx) {
    const uint32_t m = xmx_decode_min();
    return m != 0 && start_pos + 1 >= m && start_pos + 16 <= max_ctx;
}

// P4 B26: the request lanes' row step runs the G rows' decode attention as ONE launch per pass per kernel kind
// (full_attention_fa2_decode_rows: each row keeps the kernel xmx_decode / the model's default picks for it, so the bytes
// are the per-row loop's). IE_Q35_ROWS_ATTN=0 restores the per-row loop (the A/B). Read once.
inline bool rows_attn() {
    static const bool v = [] {
        const char* e = std::getenv("IE_Q35_ROWS_ATTN");
        return !(e && *e && std::atoi(e) == 0);
    }();
    return v;
}

}  // namespace ie::q35
