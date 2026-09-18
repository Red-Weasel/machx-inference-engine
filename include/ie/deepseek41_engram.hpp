// include/ie/deepseek41_engram.hpp — V4.1 engram: n-gram hash → table gather → gated residual.
//
// Read from inference/engram.py + model.py:296-368. Per position and engram layer: 24 hashed
// indices into a [384M, 256] FP8 table (6 KB of a 91.5 GB table), a [25600, 6144] FP8 Linear
// giving hc_mult keys and one value, a per-copy normalised-dot gate with a signed sqrt, and
// `h + gate * value`. The hash depends only on token ids, so every gather for a chunk is known
// before layer 0 runs.
#pragma once

#include "ie/deepseek41.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace ie {

// The tables that depend only on tokenizer + config, produced ONCE by the reference's own
// functions (tools/ds41_reference/engram_tables.py) and loaded here. Nothing in this struct is
// computed by the engine; a mismatch with the reference would silently rehash the whole table.
struct Ds41EngramTables {
    std::vector<uint32_t> layer_ids;            // e.g. {1, 14}
    uint32_t max_ngram_size = 0;                // 4
    uint32_t n_heads = 0;                       // 8
    uint32_t head_dim = 0;                      // 256
    uint32_t n_hash_cols = 0;                   // (max_ngram_size - 1) * n_heads = 24
    int64_t  pad_id = 0;                        // compressed id of the pad token
    std::vector<int32_t> token_map;             // [tokenizer vocab] -> compressed id
    // [layer][ngram-1][head], [layer][n_hash_cols], [layer][max_ngram_size]
    std::vector<std::vector<std::vector<int64_t>>> primes;
    std::vector<std::vector<int64_t>>              offsets;
    std::vector<std::vector<int64_t>>              multipliers;

    // Loads <dir>/engram_tables.json and <dir>/engram_token_map.i32. "" on success.
    std::string load(const std::string& dir);
};

// NgramHashState.forward for a prefill from position 0. An image position (token_mask False in the reference, cached
// as DEAD) is a NEGATIVE id here: no n-gram spans it.
//   out[t * n_hash_cols + j] for t in [0, L), j in [0, 24), for engram layer `layer_index`
//   (an index into `layer_ids`, NOT a model layer id).
// Integer arithmetic; must match the reference exactly.
void ds41_engram_hash(const Ds41EngramTables& tb, const int32_t* ids, uint32_t L,
                      uint32_t layer_index, int64_t* out);

// Gather + dequantise the 24 rows per position from the mmap'd table into fp32:
//   out[t][j * head_dim + d], i.e. [L, n_hash_cols * head_dim] — the flatten(-2) wkv consumes.
// `embed` is the bound Ds41Tensor pair (weight [rows, head_dim] FP8, scale [rows, head_dim/32]).
// Reads only the addressed rows; the table is never sliced or copied.
void ds41_engram_gather(const Ds41Tensor& embed, const int64_t* hash, uint32_t L,
                        uint32_t n_hash_cols, uint32_t head_dim, float* out);

}  // namespace ie
