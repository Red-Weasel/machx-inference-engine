// src/model/deepseek41_engram.cpp — V4.1 engram host side: tables, hash, gather.
#include "ie/deepseek41_engram.hpp"
#include "ie/fp8.hpp"

#include "../../third_party/nlohmann/json.hpp"

#include <fstream>

namespace ie {

using json = nlohmann::json;

std::string Ds41EngramTables::load(const std::string& dir) {
    std::ifstream f(dir + "/engram_tables.json");
    if (!f) return "cannot open " + dir + "/engram_tables.json";
    json j;
    try { j = json::parse(f); } catch (const std::exception& e) { return std::string("engram_tables.json: ") + e.what(); }

    layer_ids      = j.at("layer_ids").get<std::vector<uint32_t>>();
    max_ngram_size = j.at("max_ngram_size").get<uint32_t>();
    n_heads        = j.at("n_heads").get<uint32_t>();
    head_dim       = j.at("head_dim").get<uint32_t>();
    pad_id         = j.at("pad_id_compressed").get<int64_t>();
    primes         = j.at("primes").get<std::vector<std::vector<std::vector<int64_t>>>>();
    offsets        = j.at("offsets").get<std::vector<std::vector<int64_t>>>();
    multipliers    = j.at("multipliers").get<std::vector<std::vector<int64_t>>>();
    n_hash_cols    = (max_ngram_size - 1) * n_heads;
    const size_t n_tok = j.at("token_map_len").get<size_t>();

    std::ifstream m(dir + "/engram_token_map.i32", std::ios::binary);
    if (!m) return "cannot open " + dir + "/engram_token_map.i32";
    token_map.resize(n_tok);
    m.read(reinterpret_cast<char*>(token_map.data()), std::streamsize(n_tok * 4));
    if (size_t(m.gcount()) != n_tok * 4) return "engram_token_map.i32: short read";

    if (primes.size() != layer_ids.size() || offsets.size() != layer_ids.size() ||
        multipliers.size() != layer_ids.size())
        return "engram_tables.json: per-layer arrays disagree with layer_ids";
    for (size_t l = 0; l < layer_ids.size(); ++l) {
        if (primes[l].size() != max_ngram_size - 1 || offsets[l].size() != n_hash_cols ||
            multipliers[l].size() != max_ngram_size)
            return "engram_tables.json: layer " + std::to_string(l) + " has the wrong shape";
        for (const auto& per_ngram : primes[l])
            if (per_ngram.size() != n_heads) return "engram_tables.json: primes per head mismatch";
    }
    return {};
}

void ds41_engram_hash(const Ds41EngramTables& tb, const int32_t* ids, uint32_t L,
                      uint32_t layer_index, int64_t* out) {
    const uint32_t S = tb.max_ngram_size;
    const auto& mult = tb.multipliers[layer_index];
    const auto& prim = tb.primes[layer_index];
    const auto& offs = tb.offsets[layer_index];

    for (uint32_t t = 0; t < L; ++t) {
        // tokens[shift]: the compressed id `shift` positions back, or pad once the look-back
        // has crossed the start of the sequence. `blocked` is cumulative — once one slot is
        // blocked every further-back slot is too — matching the reference's `blocked |=`.
        int64_t prod[8];  // max_ngram_size <= 8
        bool blocked = false;
        for (uint32_t s = 0; s < S; ++s) {
            // a negative id is an image position: DEAD in the reference's cache, it ends the look-back (and its own
            // row's hash is never used: the engram gate is shut there)
            blocked = blocked || (t < s) || ids[t - s] < 0;
            const int64_t tok = blocked ? tb.pad_id : int64_t(tb.token_map[size_t(ids[t - s])]);
            prod[s] = tok * mult[s];
        }
        // rolling XOR: after step i the value is the hash of the (i+1)-gram
        int64_t rolling = prod[0];
        uint32_t col = 0;
        for (uint32_t i = 1; i < S; ++i) {
            rolling ^= prod[i];
            for (uint32_t h = 0; h < tb.n_heads; ++h, ++col)
                out[size_t(t) * tb.n_hash_cols + col] = rolling % prim[i - 1][h] + offs[col];
        }
    }
}

void ds41_engram_gather(const Ds41Tensor& embed, const int64_t* hash, uint32_t L,
                        uint32_t n_hash_cols, uint32_t head_dim, float* out) {
    const uint8_t* W = embed.w->data;
    const uint8_t* Sc = embed.s->data;
    const uint32_t sc_per_row = head_dim / 32;
    // Every (token, column) is one random 256-byte row of a 189 GB table that lives in the
    // safetensors mapping (advised MADV_RANDOM: one 4 KB fault per row, ~100 us each from the
    // NVMe). Single-threaded that was 2.6 s at pp512 and ~10 s at pp2048 per cold prompt
    // (docs/deepseek41/23, run L). The tokens are independent, so the faults go out in
    // parallel; this file is compiled with -fopenmp (src/CMakeLists.txt) so the pragma is live.
    // Phase 50 (docs/deepseek41/90): the loop runs over (token, column) PAIRS, not tokens. Over tokens, a decode step
    // (L = 1) faulted its 24 rows one after another on one thread -- 9.4 ms of a chat token's host prep, measured; the
    // pairs are independent, so every fault goes out at once. Same arithmetic per row.
    const int64_t n_pairs = int64_t(L) * int64_t(n_hash_cols);
    #pragma omp parallel for schedule(dynamic, 4)
    for (int64_t i = 0; i < n_pairs; ++i) {
        const int64_t row = hash[size_t(i)];
        const uint8_t* w = W + uint64_t(row) * head_dim;
        const uint8_t* s = Sc + uint64_t(row) * sc_per_row;
        float* o = out + size_t(i) * head_dim;
        for (uint32_t d = 0; d < head_dim; ++d)
            o[d] = e4m3_to_f32(w[d]) * e8m0_to_f32(s[d / 32]);
    }
}

}  // namespace ie
