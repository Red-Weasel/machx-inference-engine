// src/tokenizer/tokenizer.cpp — byte-level BPE for Qwen3.6.

#include "ie/tokenizer.hpp"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <ctime>
#include <limits>

namespace ie {

namespace {

// Encode a Unicode codepoint as UTF-8 (≤ 4 bytes, sufficient for codepoints up to 0x10FFFF).
std::string encode_codepoint(int cp) {
    std::string s;
    if (cp < 0x80) {
        s.push_back(char(cp));
    } else if (cp < 0x800) {
        s.push_back(char(0xC0 | (cp >> 6)));
        s.push_back(char(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        s.push_back(char(0xE0 | (cp >> 12)));
        s.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(char(0x80 | (cp & 0x3F)));
    } else {
        s.push_back(char(0xF0 | (cp >> 18)));
        s.push_back(char(0x80 | ((cp >> 12) & 0x3F)));
        s.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(char(0x80 | (cp & 0x3F)));
    }
    return s;
}

// Length (in bytes) of the UTF-8 codepoint starting at byte `c`.
inline size_t utf8_len(unsigned char c) {
    if      ((c & 0x80) == 0)    return 1;
    else if ((c & 0xE0) == 0xC0) return 2;
    else if ((c & 0xF0) == 0xE0) return 3;
    else if ((c & 0xF8) == 0xF0) return 4;
    return 1;  // invalid; treat as ASCII to make progress
}

// "Letter"-ish predicate for pretokenization. ASCII alpha + any non-ASCII byte
// (treated as letter for simplification — won't be perfectly HF-compatible
// across script boundaries but round-trips OK).
inline bool is_letter_byte(unsigned char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= 0x80);
}
inline bool is_digit_byte(unsigned char c) { return c >= '0' && c <= '9'; }
inline bool is_space_byte(unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

}  // namespace

void Tokenizer::build_byte_maps() {
    // GPT-2 byte ↔ printable codepoint convention.
    std::vector<int> bs;
    for (int b = 33;  b <= 126; ++b) bs.push_back(b);
    for (int b = 161; b <= 172; ++b) bs.push_back(b);
    for (int b = 174; b <= 255; ++b) bs.push_back(b);
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }
    for (size_t i = 0; i < bs.size(); ++i) {
        const std::string s = encode_codepoint(cs[i]);
        byte_encoder_[bs[i]] = s;
        byte_decoder_[s]     = uint8_t(bs[i]);
    }
}

std::string Tokenizer::load_from_gguf(const GgufReader& g) {
    vocab_.clear();
    vocab_lookup_.clear();
    merge_rank_.clear();
    special_ids_.clear();
    special_text_.clear();

    build_byte_maps();

    // Classic SentencePiece GGUFs (tokenizer.ggml.model=="llama": Llama-1/2,
    // Mistral, Codestral) carry SCORES, not a merges list — so the merges array
    // is required for every BPE path but ABSENT (and optional) for SPM.
    std::string tok_model;
    if (const auto* mk = g.find_kv("tokenizer.ggml.model")) tok_model = std::string(mk->as_string());
    const bool is_spm = (tok_model == "llama");

    auto* tokens_kv = g.find_kv("tokenizer.ggml.tokens");
    auto* merges_kv = g.find_kv("tokenizer.ggml.merges");
    if (!tokens_kv || tokens_kv->type != GgufValueType::kArray ||
        tokens_kv->inner_type != GgufValueType::kString) {
        return "tokenizer.ggml.tokens missing or wrong type";
    }
    const bool have_merges = merges_kv && merges_kv->type == GgufValueType::kArray &&
                             merges_kv->inner_type == GgufValueType::kString;
    if (!have_merges && !is_spm) {
        return "tokenizer.ggml.merges missing or wrong type";
    }
    auto tokens = tokens_kv->as_string_array();
    std::vector<std::string_view> merges;
    if (have_merges) merges = merges_kv->as_string_array();

    vocab_.reserve(tokens.size());
    for (auto sv : tokens) vocab_.emplace_back(sv);
    for (size_t i = 0; i < vocab_.size(); ++i) {
        vocab_lookup_.emplace(std::string_view(vocab_[i]), int32_t(i));
    }

    // token_type array — IDs marked CONTROL (==3) / USER_DEFINED (==4) are special.
    if (auto* tt = g.find_kv("tokenizer.ggml.token_type")) {
        if (tt->type == GgufValueType::kArray && tt->inner_type == GgufValueType::kI32) {
            auto types = tt->as_pod_array<int32_t>();
            for (size_t i = 0; i < types.size() && i < vocab_.size(); ++i) {
                // ggml: 1=NORMAL, 2=UNKNOWN, 3=CONTROL, 4=USER_DEFINED, 5=UNUSED, 6=BYTE.
                if (types[i] == 3 || types[i] == 4) {
                    special_ids_.insert(int32_t(i));
                    special_text_.emplace_back(vocab_[i], int32_t(i));
                }
            }
        }
    }
    // Sort longest-first so greedy match prefers longer specials.
    std::sort(special_text_.begin(), special_text_.end(),
              [](const auto& a, const auto& b){ return a.first.size() > b.first.size(); });

    // Parse merges. Each entry is "tokA tokB" where tokA, tokB are vocab strings.
    merge_rank_.reserve(merges.size());
    for (size_t r = 0; r < merges.size(); ++r) {
        std::string_view m = merges[r];
        const auto sp = m.find(' ');
        if (sp == std::string_view::npos) continue;
        const auto a_str = m.substr(0, sp);
        const auto b_str = m.substr(sp + 1);
        const auto a_it = vocab_lookup_.find(a_str);
        const auto b_it = vocab_lookup_.find(b_str);
        if (a_it == vocab_lookup_.end() || b_it == vocab_lookup_.end()) continue;
        const uint64_t key = (uint64_t(uint32_t(a_it->second)) << 32) |
                              uint64_t(uint32_t(b_it->second));
        merge_rank_.emplace(key, int32_t(r));
    }

    if (auto* kv = g.find_kv("tokenizer.ggml.bos_token_id"))     bos_id_     = int32_t(kv->as_int());
    if (auto* kv = g.find_kv("tokenizer.ggml.eos_token_id"))     eos_id_     = int32_t(kv->as_int());
    if (auto* kv = g.find_kv("tokenizer.ggml.padding_token_id")) pad_id_     = int32_t(kv->as_int());
    if (auto* kv = g.find_kv("tokenizer.ggml.add_bos_token"))    add_bos_token_ = kv->as_bool();

    // P3a / Wave-1: pre-type dispatch. "llama-bpe" → digit triplets +
    // ignore_merges. "tekken" → case-aware letter split + single digits +
    // ignore_merges (see pretokenize_tekken + llama.cpp LLAMA_VOCAB_PRE_TYPE_TEKKEN).
    if (const auto* kv = g.find_kv("tokenizer.ggml.pre")) pre_ = std::string(kv->as_string());
    tekken_        = (pre_ == "tekken");
    // tekken splits digits singly (\p{N}); llama-bpe and dbrx group runs of 1-3.
    // dbrx (Phi-4, cl100k/gpt2 BPE) pretokenizes IDENTICALLY to llama-bpe — the
    // only split delta vs the qwen2 default is digit-runs-≤3 (which digits_1to3_
    // gives) + ignore_merges; it differs only in vocab/merges (from the GGUF).
    digits_1to3_   = (pre_ == "llama-bpe") || (pre_ == "dbrx");
    // tekken sets ignore_merges=true in llama.cpp (llama-vocab.cpp); llama-bpe + dbrx too.
    ignore_merges_ = (pre_ == "llama-bpe") || (pre_ == "tekken") || (pre_ == "dbrx");
    // llama-bpe / tekken GGUFs may omit add_bos_token, but the model REQUIRES
    // BOS (<|begin_of_text|> / <s>) prepended → default it true for those families.
    // (Mistral tekken GGUFs carry add_bos_token=true, so this is a safety default.)
    if (!g.find_kv("tokenizer.ggml.add_bos_token") && (pre_ == "llama-bpe" || pre_ == "tekken"))
        add_bos_token_ = true;
    // Gemma 4: SPM-style raw-UTF-8 BPE (spaces→U+2581, split on \n, <0xXX> byte
    // fallback). Detected via tokenizer.ggml.model=="gemma4" (the GGUF omits
    // tokenizer.ggml.pre; llama derives pre from the model). The model prepends
    // BOS (llama-tokenize default) even though the GGUF sets add_bos_token=false.
    gemma_ = (tok_model == "gemma4");
    if (gemma_) add_bos_token_ = true;

    // Classic SentencePiece (model=="llama"): merge by token SCORE, leading dummy
    // ▁ (add_space_prefix, default true), <0xXX> byte fallback. Mistral/Codestral
    // GGUFs carry add_bos_token=true (read above); default it for robustness.
    spm_ = is_spm;
    if (spm_) {
        if (const auto* sk = g.find_kv("tokenizer.ggml.scores");
            sk && sk->type == GgufValueType::kArray && sk->inner_type == GgufValueType::kF32) {
            auto sc = sk->as_pod_array<float>();
            scores_.assign(sc.begin(), sc.end());
        }
        if (scores_.size() != vocab_.size())
            return "SPM tokenizer: tokenizer.ggml.scores missing or size != vocab";
        if (const auto* kv = g.find_kv("tokenizer.ggml.add_space_prefix"))
            add_space_prefix_ = kv->as_bool();
        if (!g.find_kv("tokenizer.ggml.add_bos_token")) add_bos_token_ = true;
    }
    return {};
}

bool Tokenizer::is_special(int32_t id) const noexcept {
    return special_ids_.count(id) > 0;
}
std::string_view Tokenizer::token_str(int32_t id) const noexcept {
    if (id < 0 || size_t(id) >= vocab_.size()) return {};
    return vocab_[id];
}
int32_t Tokenizer::find_token(std::string_view s) const noexcept {
    auto it = vocab_lookup_.find(s);
    return it == vocab_lookup_.end() ? -1 : it->second;
}

int32_t Tokenizer::bpe_lookup_or_neg(std::string_view s) const noexcept {
    auto it = vocab_lookup_.find(s);
    return it == vocab_lookup_.end() ? -1 : it->second;
}

// Apply BPE to a byte-encoded "word" (after passing each byte through byte_encoder_).
// Returns a list of token IDs.
std::vector<int32_t> Tokenizer::bpe_merge_word(std::string_view encoded) const {
    std::vector<int32_t> ids;
    if (encoded.empty()) return ids;

    // Initial split: each Unicode codepoint = one token (every printable char in
    // the byte_encoder map has a vocab entry).
    for (size_t i = 0; i < encoded.size();) {
        const size_t n = utf8_len(uint8_t(encoded[i]));
        const std::string_view ch = encoded.substr(i, n);
        const int32_t id = bpe_lookup_or_neg(ch);
        if (id < 0) {
            // Should not happen for byte-encoded input — fall back: emit unknown.
            ids.push_back(-1);
        } else {
            ids.push_back(id);
        }
        i += n;
    }

    // Iteratively merge the lowest-rank pair until none remains.
    while (ids.size() > 1) {
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        size_t  best_i    = 0;
        for (size_t i = 0; i + 1 < ids.size(); ++i) {
            if (ids[i] < 0 || ids[i + 1] < 0) continue;
            const uint64_t key = (uint64_t(uint32_t(ids[i])) << 32) |
                                  uint64_t(uint32_t(ids[i + 1]));
            const auto it = merge_rank_.find(key);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i    = i;
            }
        }
        if (best_rank == std::numeric_limits<int32_t>::max()) break;
        // Merge ids[best_i] and ids[best_i+1].
        const std::string merged_str =
            std::string(vocab_[ids[best_i]]) + std::string(vocab_[ids[best_i + 1]]);
        const int32_t merged_id = bpe_lookup_or_neg(merged_str);
        if (merged_id < 0) break;
        ids[best_i] = merged_id;
        ids.erase(ids.begin() + best_i + 1);
    }
    return ids;
}

// Gemma SPM-style BPE on a raw-UTF-8 segment (spaces already U+2581; no \n).
// Initial symbols are UTF-8 codepoints (NOT byte-encoded); merge by lowest rank;
// resolve to ids with <0xXX> byte fallback for anything not in the vocab.
void Tokenizer::bpe_merge_gemma(std::string_view seg, std::vector<int32_t>& out) const {
    if (seg.empty()) return;
    struct Sym { std::string s; int32_t id; };
    std::vector<Sym> syms;
    for (size_t i = 0; i < seg.size();) {
        const size_t n = utf8_len(uint8_t(seg[i]));
        std::string ch(seg.substr(i, std::min(n, seg.size() - i)));
        const int32_t id = bpe_lookup_or_neg(ch);
        syms.push_back({std::move(ch), id});
        i += n;
    }
    while (syms.size() > 1) {
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        size_t  best_i = 0;
        for (size_t i = 0; i + 1 < syms.size(); ++i) {
            if (syms[i].id < 0 || syms[i + 1].id < 0) continue;
            const uint64_t key = (uint64_t(uint32_t(syms[i].id)) << 32) |
                                  uint64_t(uint32_t(syms[i + 1].id));
            const auto it = merge_rank_.find(key);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second; best_i = i;
            }
        }
        if (best_rank == std::numeric_limits<int32_t>::max()) break;
        std::string merged = syms[best_i].s + syms[best_i + 1].s;
        const int32_t mid = bpe_lookup_or_neg(merged);
        if (mid < 0) break;
        syms[best_i] = {std::move(merged), mid};
        syms.erase(syms.begin() + best_i + 1);
    }
    static const char* kHex = "0123456789ABCDEF";
    for (auto& sym : syms) {
        if (sym.id >= 0) { out.push_back(sym.id); continue; }
        for (unsigned char b : sym.s) {                       // byte fallback <0xXX>
            const char buf[7] = {'<', '0', 'x', kHex[b >> 4], kHex[b & 15], '>', 0};
            const int32_t bid = bpe_lookup_or_neg(std::string_view(buf, 6));
            if (bid >= 0) out.push_back(bid);
        }
    }
}

// Classic SentencePiece BPE (Llama-1/2, Mistral, Codestral) on a raw-UTF-8
// segment (spaces already U+2581 by the caller, no regex pre-split). Mirrors
// llama.cpp llm_tokenizer_spm: initial symbols are UTF-8 codepoints; each step
// merges the adjacent pair whose MERGED token has the highest score (leftmost
// on ties); unresolved symbols fall back to <0xXX> byte tokens. The greedy
// "merge the global-best pair each pass" is equivalent to llama's priority
// queue (both repeatedly merge the highest-scoring mergeable bigram).
void Tokenizer::bpe_merge_spm(std::string_view seg, std::vector<int32_t>& out) const {
    if (seg.empty()) return;
    struct Sym { std::string s; int32_t id; };
    std::vector<Sym> syms;
    for (size_t i = 0; i < seg.size();) {
        const size_t n = utf8_len(uint8_t(seg[i]));
        std::string ch(seg.substr(i, std::min(n, seg.size() - i)));
        const int32_t id = bpe_lookup_or_neg(ch);
        syms.push_back({std::move(ch), id});
        i += n;
    }
    while (syms.size() > 1) {
        float   best_score = -std::numeric_limits<float>::infinity();
        size_t  best_i     = std::numeric_limits<size_t>::max();
        int32_t best_id    = -1;
        for (size_t i = 0; i + 1 < syms.size(); ++i) {
            const std::string merged = syms[i].s + syms[i + 1].s;
            const int32_t mid = bpe_lookup_or_neg(merged);
            if (mid < 0) continue;
            const float sc = scores_[mid];
            if (sc > best_score) { best_score = sc; best_i = i; best_id = mid; }
        }
        if (best_i == std::numeric_limits<size_t>::max()) break;
        syms[best_i] = {syms[best_i].s + syms[best_i + 1].s, best_id};
        syms.erase(syms.begin() + best_i + 1);
    }
    static const char* kHex = "0123456789ABCDEF";
    for (auto& sym : syms) {
        if (sym.id >= 0) { out.push_back(sym.id); continue; }
        for (unsigned char b : sym.s) {                       // byte fallback <0xXX>
            const char buf[7] = {'<', '0', 'x', kHex[b >> 4], kHex[b & 15], '>', 0};
            const int32_t bid = bpe_lookup_or_neg(std::string_view(buf, 6));
            if (bid >= 0) out.push_back(bid);
        }
    }
}

// Pretokenize: split text into "words" per the qwen2 pre-tokenizer regex
// (tokenizer.json / llama.cpp LLAMA_VOCAB_PRE_TYPE_QWEN2), alternatives tried
// in order at each position:
//   alt 1: `(?i:'s|'t|'re|'ve|'m|'ll|'d)`   — contractions, case-insensitive
//   alt 2: `[^\r\n\p{L}\p{N}]?\p{L}+`       — letters, optional 1-char prefix
//   alt 3: `\p{N}`                          — single digit (NB: per char)
//   alt 4: ` ?[^\s\p{L}\p{N}]+[\r\n]*`      — punct+, opt space, CR/LF tail
//   alt 5: `\s*[\r\n]+`                     — whitespace through last newline
//   alt 6: `\s+(?!\S)`                      — trailing ws / run minus last
//   alt 7: `\s+`                            — whitespace fallback
//
// Simplified Unicode handling: any byte ≥ 0x80 is treated as a "letter" byte,
// so CJK and emoji match alt 2. This is a known approximation — full
// `\p{L}\p{N}` Unicode classification is in the Phase 9 backlog.
static std::vector<std::string_view> pretokenize_simple(std::string_view text,
                                                        bool digits_1to3) {
    std::vector<std::string_view> out;
    const size_t n = text.size();
    const auto is_crlf = [](unsigned char x) { return x == '\r' || x == '\n'; };
    size_t i = 0;
    while (i < n) {
        const unsigned char c = uint8_t(text[i]);

        // ---- alt 1: contraction (?i:'s|'t|'re|'ve|'m|'ll|'d) ----
        // No word-boundary in the regex: matches anywhere an apostrophe is
        // followed by a contraction suffix (so `'sup` -> `'s` + `up`).
        if (c == '\'' && i + 1 < n) {
            const unsigned char a = uint8_t(text[i + 1]) | 0x20;  // ASCII lower
            size_t len = 0;
            if (a == 's' || a == 't' || a == 'm' || a == 'd') {
                len = 2;
            } else if (i + 2 < n) {
                const unsigned char b = uint8_t(text[i + 2]) | 0x20;
                if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') ||
                    (a == 'l' && b == 'l')) len = 3;
            }
            if (len) { out.emplace_back(text.substr(i, len)); i += len; continue; }
        }

        // ---- alt 2: [^\r\n\p{L}\p{N}]?\p{L}+ ----
        // The optional prefix is any single non-letter/digit/CR/LF char
        // (space, apostrophe, `.`/`/`, ...), so `O'Brien` -> `O` + `'Brien`
        // and `console.log` -> `console` + `.log`. Prefix is 1 byte here:
        // every byte >= 0x80 already counts as a letter.
        {
            size_t j = i;
            if (!is_letter_byte(c) && !is_digit_byte(c) && !is_crlf(c)) ++j;
            if (j < n && is_letter_byte(uint8_t(text[j]))) {
                while (j < n && is_letter_byte(uint8_t(text[j])))
                    j += utf8_len(uint8_t(text[j]));
                out.emplace_back(text.substr(i, j - i));
                i = j;
                continue;
            }
        }

        // ---- alt 3: \p{N} (qwen2, single) vs \p{N}{1,3} (llama-bpe, run≤3) ----
        if (is_digit_byte(c)) {
            size_t len = 1;
            if (digits_1to3)
                while (len < 3 && i + len < n && is_digit_byte(uint8_t(text[i + len]))) ++len;
            out.emplace_back(text.substr(i, len)); i += len; continue;
        }

        // ---- alt 4: ` ?[^\s\p{L}\p{N}]+[\r\n]*` ----
        {
            size_t j = i;
            if (text[j] == ' ') ++j;            // optional literal space
            if (j < n) {
                const unsigned char d = uint8_t(text[j]);
                if (!is_space_byte(d) && !is_letter_byte(d) && !is_digit_byte(d)) {
                    while (j < n) {
                        const unsigned char x = uint8_t(text[j]);
                        if (is_space_byte(x) || is_letter_byte(x) || is_digit_byte(x)) break;
                        j += utf8_len(x);
                    }
                    while (j < n && is_crlf(uint8_t(text[j]))) ++j;  // [\r\n]*
                    out.emplace_back(text.substr(i, j - i));
                    i = j;
                    continue;
                }
            }
        }

        // ---- alts 5/6/7: whitespace ----
        if (is_space_byte(c)) {
            size_t e = i;
            size_t last_nl = std::string_view::npos;
            while (e < n && is_space_byte(uint8_t(text[e]))) {
                if (is_crlf(uint8_t(text[e]))) last_nl = e;
                ++e;
            }
            if (last_nl != std::string_view::npos) {
                // alt 5: `\s*[\r\n]+` — backtracking yields the span through
                // the last CR/LF in the run; trailing spaces/tabs re-enter the
                // loop (alt 2 attach / alt 6/7).
                out.emplace_back(text.substr(i, last_nl + 1 - i));
                i = last_nl + 1;
            } else if (e == n) {
                // alt 6: trailing whitespace — `(?!\S)` holds at EOS.
                out.emplace_back(text.substr(i, e - i));
                i = e;
            } else if (e - i > 1) {
                // alt 6: run minus last char; the final space is left for the
                // next match's optional-prefix (alt 2) or ` ?` (alt 4).
                out.emplace_back(text.substr(i, e - i - 1));
                i = e - 1;
            } else {
                // alt 7: single whitespace char before a non-space that no
                // earlier alternative claimed (e.g. ` 5`, `\t!`).
                out.emplace_back(text.substr(i, 1));
                ++i;
            }
            continue;
        }

        // Unreachable for well-formed input — emit one byte to guarantee progress.
        out.emplace_back(text.substr(i, 1));
        ++i;
    }
    return out;
}

// Wave-1: tekken pre-tokenizer (Mistral Nemo / Small-24B / Devstral / Codestral).
// Mirrors llama.cpp LLAMA_VOCAB_PRE_TYPE_TEKKEN (llama-vocab.cpp). Original
// tokenizer.json regex (the authoritative spec), alternatives tried in order:
//   altA: [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+
//   altB: [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*
//   altN: \p{N}                              — SINGLE digit (NOT runs of 3)
//   altP: ` ?[^\s\p{L}\p{N}]+[\r\n/]*`       — punct+, opt space, CR/LF/`/` tail
//   altW: `\s*[\r\n]+` | `\s+(?!\S)` | `\s+` — whitespace (same as qwen2)
//
// altA+altB collapse to a CASE-AWARE letter word: an optional 1-char non-letter
// prefix, then a run of "uppercase-ish" letters, then a run of "lowercase-ish"
// letters — so `camelCase`→`camel`+`Case`, `XMLHttp`→`XMLHttp`, `iPhone`→`i`+
// `Phone` (the property that makes tekken correct on CODE identifiers). The set
// \p{Lm}\p{Lo}\p{M} belongs to BOTH case classes, so a non-ASCII letter byte
// never forces a case boundary (it extends whichever run it's in).
//
// Unicode approximation (same as pretokenize_simple, documented Phase-9 backlog):
// any byte >= 0x80 is treated as a both-case letter. This is exact for ASCII +
// common Latin (incl. accented prose) and ALL code; it diverges only on emoji /
// non-letter symbols (\p{So}), which the engine misclassifies as letters
// engine-wide. Verified bit-exact vs the reference tekken regex on 50k random
// ASCII+code strings and the prose+code golden corpus (tekken_tokenizer_test).
static std::vector<std::string_view> pretokenize_tekken(std::string_view text) {
    std::vector<std::string_view> out;
    const size_t n = text.size();
    const auto is_crlf  = [](unsigned char x) { return x == '\r' || x == '\n'; };
    const auto is_upper = [](unsigned char x) { return x >= 'A' && x <= 'Z'; };
    const auto is_lower = [](unsigned char x) { return x >= 'a' && x <= 'z'; };
    size_t i = 0;
    while (i < n) {
        const unsigned char c = uint8_t(text[i]);

        // ---- altA/altB: case-aware letter word (with optional 1-char prefix) ----
        {
            size_t j = i;
            if (!is_letter_byte(c) && !is_digit_byte(c) && !is_crlf(c) &&
                i + 1 < n && is_letter_byte(uint8_t(text[i + 1]))) {
                j = i + 1;                 // optional non-letter/digit prefix
            }
            if (j < n && is_letter_byte(uint8_t(text[j]))) {
                size_t k = j;
                // phase 1: uppercase (and non-ASCII both-case) letters.
                while (k < n && is_letter_byte(uint8_t(text[k])) && !is_lower(uint8_t(text[k])))
                    k += utf8_len(uint8_t(text[k]));
                // phase 2: lowercase (and non-ASCII both-case) letters.
                while (k < n && is_letter_byte(uint8_t(text[k])) && !is_upper(uint8_t(text[k])))
                    k += utf8_len(uint8_t(text[k]));
                if (k > j) { out.emplace_back(text.substr(i, k - i)); i = k; continue; }
            }
        }

        // ---- altN: \p{N} — SINGLE digit ----
        if (is_digit_byte(c)) { out.emplace_back(text.substr(i, 1)); ++i; continue; }

        // ---- altP: ` ?[^\s\p{L}\p{N}]+[\r\n/]*` ----
        {
            size_t j = i;
            if (text[j] == ' ') ++j;            // optional literal space
            if (j < n) {
                const unsigned char d = uint8_t(text[j]);
                if (!is_space_byte(d) && !is_letter_byte(d) && !is_digit_byte(d)) {
                    while (j < n) {
                        const unsigned char x = uint8_t(text[j]);
                        if (is_space_byte(x) || is_letter_byte(x) || is_digit_byte(x)) break;
                        j += utf8_len(x);
                    }
                    // tekken tail is [\r\n/]* (NOTE the `/`, unlike qwen2's [\r\n]*).
                    while (j < n && (is_crlf(uint8_t(text[j])) || text[j] == '/')) ++j;
                    out.emplace_back(text.substr(i, j - i));
                    i = j;
                    continue;
                }
            }
        }

        // ---- altW: whitespace (identical to qwen2 alts 5/6/7) ----
        if (is_space_byte(c)) {
            size_t e = i;
            size_t last_nl = std::string_view::npos;
            while (e < n && is_space_byte(uint8_t(text[e]))) {
                if (is_crlf(uint8_t(text[e]))) last_nl = e;
                ++e;
            }
            if (last_nl != std::string_view::npos) {
                out.emplace_back(text.substr(i, last_nl + 1 - i));
                i = last_nl + 1;
            } else if (e == n) {
                out.emplace_back(text.substr(i, e - i));
                i = e;
            } else if (e - i > 1) {
                out.emplace_back(text.substr(i, e - i - 1));
                i = e - 1;
            } else {
                out.emplace_back(text.substr(i, 1));
                ++i;
            }
            continue;
        }

        // Unreachable for well-formed input — emit one byte to guarantee progress.
        out.emplace_back(text.substr(i, 1));
        ++i;
    }
    return out;
}

std::vector<int32_t> Tokenizer::encode(std::string_view text, bool allow_special) const {
    std::vector<int32_t> out;
    if (vocab_.empty()) return out;
    if (add_bos_token_ && bos_id_ >= 0) out.push_back(bos_id_);

    // Greedy split around special-token literals first.
    auto encode_chunk = [&](std::string_view chunk) {
        if (chunk.empty()) return;
        if (gemma_) {
            // SPM normalize: replace each space with U+2581; no leading prefix
            // (add_space_prefix=false). Then BPE over the whole text, split only
            // on newlines (the gemma4 pre regex is [^\n]+|[\n]+).
            std::string norm;
            norm.reserve(chunk.size() + chunk.size() / 2);
            for (char c : chunk) {
                if (c == ' ') norm += "\xe2\x96\x81";   // U+2581 ▁
                else          norm += c;
            }
            size_t i = 0;
            while (i < norm.size()) {
                const bool nl = (norm[i] == '\n');
                size_t j = i;
                while (j < norm.size() && (norm[j] == '\n') == nl) ++j;
                bpe_merge_gemma(std::string_view(norm).substr(i, j - i), out);
                i = j;
            }
            return;
        }
        if (spm_) {
            // Classic SPM: a leading dummy ▁ (add_space_prefix; every text
            // fragment follows BOS or a special, so is_prev_special always holds),
            // then escape spaces→U+2581 and merge the whole chunk by score.
            std::string norm;
            norm.reserve(chunk.size() + chunk.size() / 2 + 3);
            if (add_space_prefix_) norm += "\xe2\x96\x81";   // U+2581 ▁
            for (char c : chunk) {
                if (c == ' ') norm += "\xe2\x96\x81";
                else          norm += c;
            }
            bpe_merge_spm(norm, out);
            return;
        }
        // Split into pretokens. tekken uses a distinct case-aware split; all
        // other pre-types (default/qwen2/llama-bpe) use the shared simple path.
        const auto pretoks = tekken_ ? pretokenize_tekken(chunk)
                                      : pretokenize_simple(chunk, digits_1to3_);
        for (auto pt : pretoks) {
            // Byte-encode each byte of the pretoken.
            std::string enc;
            enc.reserve(pt.size() * 2);
            for (char c : pt) enc += byte_encoder_[uint8_t(c)];
            // P3a: ignore_merges (llama-bpe) — if the whole byte-encoded
            // pretoken is itself a vocab token, emit it directly and skip BPE
            // (llama-vocab.cpp:577). Otherwise fall through to the merge loop.
            if (ignore_merges_) {
                const int32_t whole = bpe_lookup_or_neg(enc);
                if (whole >= 0) { out.push_back(whole); continue; }
            }
            // BPE merge.
            const auto ids = bpe_merge_word(enc);
            for (auto id : ids) {
                if (id >= 0) out.push_back(id);
            }
        }
    };

    if (!allow_special || special_text_.empty()) {
        encode_chunk(text);
        return out;
    }
    size_t i = 0;
    while (i < text.size()) {
        // Try to match a special token at the current position.
        bool matched = false;
        for (const auto& [s, id] : special_text_) {
            if (i + s.size() <= text.size() &&
                std::memcmp(text.data() + i, s.data(), s.size()) == 0) {
                // Encode the prefix before this special, then emit the special.
                encode_chunk(text.substr(0, 0));   // no-op; placeholder
                out.push_back(id);
                i += s.size();
                matched = true;
                break;
            }
        }
        if (matched) continue;
        // Find the start of the next special token (or end).
        size_t next = text.size();
        for (const auto& [s, _] : special_text_) {
            const auto pos = text.find(s, i);
            if (pos != std::string_view::npos && pos < next) next = pos;
        }
        encode_chunk(text.substr(i, next - i));
        i = next;
    }
    return out;
}

std::string Tokenizer::decode(std::span<const int32_t> ids, bool skip_special,
                              std::span<const int32_t> keep_special) const {
    std::string out;
    out.reserve(ids.size() * 4);
    for (int32_t id : ids) {
        if (id < 0 || size_t(id) >= vocab_.size()) continue;
        if (is_special(id)) {
            const bool keep = std::find(keep_special.begin(), keep_special.end(), id)
                              != keep_special.end();
            if (skip_special && !keep) continue;
            out += vocab_[id];           // emit literal special-token text
            continue;
        }
        const std::string_view t = vocab_[id];
        if (spm_) {
            // SPM tokens are raw UTF-8 (not GPT-2 byte-encoded): a <0xXX> token
            // decodes to that single byte (consecutive byte tokens reassemble a
            // multibyte codepoint); U+2581 (▁) decodes to a space.
            if (t.size() == 6 && t[0] == '<' && t[1] == '0' && t[2] == 'x' && t[5] == '>') {
                const auto hx = [](char c) -> int {
                    if (c >= '0' && c <= '9') return c - '0';
                    c |= 0x20; return (c >= 'a' && c <= 'f') ? c - 'a' + 10 : -1;
                };
                const int hi = hx(t[3]), lo = hx(t[4]);
                if (hi >= 0 && lo >= 0) { out.push_back(char((hi << 4) | lo)); continue; }
            }
            for (size_t i = 0; i < t.size();) {
                if (i + 3 <= t.size() && uint8_t(t[i]) == 0xE2 &&
                    uint8_t(t[i + 1]) == 0x96 && uint8_t(t[i + 2]) == 0x81) {
                    out.push_back(' '); i += 3;
                } else { out.push_back(t[i]); ++i; }
            }
            continue;
        }
        // Byte-decode each codepoint of the vocab string.
        for (size_t i = 0; i < t.size();) {
            const size_t n = utf8_len(uint8_t(t[i]));
            const std::string_view ch = t.substr(i, n);
            const auto it = byte_decoder_.find(std::string(ch));
            if (it != byte_decoder_.end()) {
                out.push_back(char(it->second));
            } else {
                // Unknown codepoint — emit raw (shouldn't happen for valid vocab).
                out.append(ch);
            }
            i += n;
        }
    }
    return out;
}

// ===== Chat template =====

// Canonical Qwen3 tools preamble (JSON tool-call convention). Chosen over the
// Qwen3.6-native <function=NAME> XML format (research/04 §4.2) because
// OpenAI-compat clients (Seal) recover text-embedded tool calls from the JSON
// `{"name":..., "arguments":...}` shape — see docs/seal-integration.md gap 1.
static void append_tools_preamble(std::string& s, std::string_view tools_json) {
    s += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
         "You are provided with function signatures within <tools></tools> XML tags:\n"
         "<tools>";
    // One JSON schema object per line, client key order preserved.
    nlohmann::ordered_json tools = nlohmann::ordered_json::parse(
        tools_json, nullptr, /*allow_exceptions=*/false);
    if (tools.is_array()) {
        for (const auto& t : tools) { s += '\n'; s += t.dump(); }
    } else {
        s += '\n'; s += tools_json;   // unparseable: embed verbatim
    }
    s += "\n</tools>\n\n"
         "For each function call, return a json object with function name and "
         "arguments within <tool_call></tool_call> XML tags:\n"
         "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";
}

std::string build_chatml_prompt(std::span<const ChatTurn> turns,
                                bool add_generation_prompt,
                                bool enable_thinking,
                                std::string_view tools_json,
                                bool model_has_think) {
    std::string s;
    size_t i = 0;
    const size_t n = turns.size();
    if (!tools_json.empty()) {
        // Tools preamble lives in the system turn (created if absent).
        s += "<|im_start|>system\n";
        if (n > 0 && turns[0].role == "system") {
            if (!turns[0].content.empty()) { s += turns[0].content; s += "\n\n"; }
            i = 1;
        }
        append_tools_preamble(s, tools_json);
        s += "<|im_end|>\n";
    }
    for (; i < n; ++i) {
        const auto& t = turns[i];
        if (t.role == "tool") {
            // Tool results render inside a user turn as <tool_response> blocks;
            // consecutive tool turns merge into one user turn (Qwen3 template).
            s += "<|im_start|>user";
            while (i < n && turns[i].role == "tool") {
                s += "\n<tool_response>\n";
                s += turns[i].content;
                s += "\n</tool_response>";
                ++i;
            }
            --i;                       // for-loop will ++i past the last tool turn
            s += "<|im_end|>\n";
            continue;
        }
        s += "<|im_start|>";
        s += t.role;
        s += '\n';
        s += t.content;
        s += "<|im_end|>\n";
    }
    if (add_generation_prompt) {
        s += "<|im_start|>assistant\n";
        if (model_has_think) {
            if (enable_thinking) {
                s += "<think>\n";
            } else {
                // Qwen3-family convention: an empty think block signals "reasoning
                // already done" and suppresses the reasoning trace.
                s += "<think>\n\n</think>\n\n";
            }
        }
        // Non-thinking models (Qwen3-Coder): plain ChatML, no <think> block.
    }
    return s;
}

std::string build_llama3_prompt(std::span<const ChatTurn> turns,
                                bool add_generation_prompt) {
    // <|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>
    // BOS is added by encode() (add_bos_token=true); not a template literal.
    std::string s;
    for (const auto& t : turns) {
        s += "<|start_header_id|>";
        s += t.role;                       // "system" | "user" | "assistant"
        s += "<|end_header_id|>\n\n";
        s += t.content;
        s += "<|eot_id|>";
    }
    if (add_generation_prompt)
        s += "<|start_header_id|>assistant<|end_header_id|>\n\n";
    return s;
}

std::string build_mistral_prompt(std::span<const ChatTurn> turns,
                                 bool add_generation_prompt,
                                 bool system_prompt_block) {
    // <s>[INST] {user} [/INST] {assistant}</s>[INST] {u2} [/INST] ...
    // BOS (<s>) is added by encode() (add_bos_token), not a template literal.
    std::string s;
    size_t i = 0;
    const size_t n = turns.size();
    std::string sys;
    if (n > 0 && turns[0].role == "system") { sys = turns[0].content; i = 1; }
    bool first_user = true;
    for (; i < n; ++i) {
        const auto& t = turns[i];
        if (t.role == "user") {
            s += "[INST] ";
            if (first_user && !sys.empty()) {
                if (system_prompt_block) {
                    s += "[SYSTEM_PROMPT]"; s += sys; s += "[/SYSTEM_PROMPT]\n\n";
                } else {
                    s += sys; s += "\n\n";
                }
            }
            s += t.content;
            s += " [/INST]";
            first_user = false;
        } else if (t.role == "assistant") {
            s += ' ';
            s += t.content;
            s += "</s>";
        }
    }
    // No trailing token needed for generation: the model continues after [/INST].
    (void)add_generation_prompt;
    return s;
}

std::string build_deepseek_prompt(std::span<const ChatTurn> turns,
                                  bool add_generation_prompt,
                                  bool enable_thinking) {
    // {system}<｜User｜>{u}<｜Assistant｜>{a}<｜end▁of▁sentence｜>...   (R1-Distill)
    // The fullwidth sentinels (U+FF5C bar, U+2581 underscore) ship as special
    // tokens in the GGUF; encode() greedily matches them. BOS
    // (<｜begin▁of▁sentence｜>) is prepended by encode() (add_bos_token).
    // Split string literals so a hex escape never runs into a following hex
    // digit ('A','e','f' etc. are hex digits): "\x9c" "User" not "\x9cUser".
    static constexpr const char* kUser      = "<\xef\xbd\x9c" "User" "\xef\xbd\x9c>";
    static constexpr const char* kAssistant = "<\xef\xbd\x9c" "Assistant" "\xef\xbd\x9c>";
    static constexpr const char* kEos       =
        "<\xef\xbd\x9c" "end" "\xe2\x96\x81" "of" "\xe2\x96\x81" "sentence" "\xef\xbd\x9c>";
    std::string s;
    size_t i = 0;
    const size_t n = turns.size();
    if (n > 0 && turns[0].role == "system") { s += turns[0].content; i = 1; }
    for (; i < n; ++i) {
        const auto& t = turns[i];
        if (t.role == "user") {
            s += kUser;
            s += t.content;
        } else if (t.role == "assistant") {
            s += kAssistant;
            s += t.content;
            s += kEos;
        }
    }
    if (add_generation_prompt) {
        s += kAssistant;
        // R1-Distill emits a <think> reasoning trace; mirror the ChatML gate so
        // enable_thinking=false yields the empty-think convention.
        if (enable_thinking) s += "<think>\n";
        else                 s += "<think>\n\n</think>\n\n";
    }
    return s;
}

std::string build_granite_prompt(std::span<const ChatTurn> turns,
                                 bool add_generation_prompt) {
    // IBM Granite-3.x:
    //   <|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n  (per turn)
    //   <|start_of_role|>assistant<|end_of_role|>                         (gen prompt)
    // The three markers ship as single special tokens (49152/49153/0); encode()
    // greedily matches them. add_bos_token=false → encode() prepends no BOS.
    // Assistant turns close on <|end_of_text|> (=eos id 0, caught at sampling).
    // v1: the convert-time default system message (knowledge-cutoff + today's
    // date) and tool rendering are NOT injected — bare turns chat coherently.
    static constexpr const char* kSOR = "<|start_of_role|>";
    static constexpr const char* kEOR = "<|end_of_role|>";
    static constexpr const char* kEOT = "<|end_of_text|>";
    std::string s;
    for (const auto& t : turns) {
        s += kSOR; s += t.role; s += kEOR;
        s += t.content;
        s += kEOT;
        s += "\n";
    }
    if (add_generation_prompt) {
        s += kSOR; s += "assistant"; s += kEOR;
    }
    return s;
}

std::string build_gemma_prompt(std::span<const ChatTurn> turns,
                               bool add_generation_prompt) {
    // Gemma 4: <|turn>{role}\n{content}<turn|>\n per turn (assistant→model),
    // <|turn>model\n to start generation. <|turn>=105, <turn|>=106 are single
    // special tokens; BOS prepended by encode(); the model closes with <turn|>.
    std::string s;
    for (const auto& t : turns) {
        const std::string role = (t.role == "assistant") ? "model" : t.role;
        s += "<|turn>"; s += role; s += "\n";
        s += t.content;
        s += "<turn|>\n";
    }
    if (add_generation_prompt) s += "<|turn>model\n";
    return s;
}

// JSON-Schema (common subset) -> a TypeScript type string for a Harmony tool
// declaration. Covers string/number/integer/boolean/array/object/enum; anything
// else degrades to `any` (still a valid signature the model can call).
static std::string harmony_schema_to_ts(const nlohmann::ordered_json& sch) {
    if (!sch.is_object()) return "any";
    if (sch.contains("enum") && sch["enum"].is_array() && !sch["enum"].empty()) {
        std::string u;
        for (const auto& e : sch["enum"]) {
            if (!u.empty()) u += " | ";
            u += e.is_string() ? ("\"" + e.get<std::string>() + "\"") : e.dump();
        }
        return u;
    }
    const std::string type = sch.value("type", std::string());
    if (type == "string")  return "string";
    if (type == "number" || type == "integer") return "number";
    if (type == "boolean") return "boolean";
    if (type == "array")
        return (sch.contains("items") ? harmony_schema_to_ts(sch["items"]) : std::string("any")) + "[]";
    if (type == "object" || sch.contains("properties")) {
        std::string o = "{ ";
        if (sch.contains("properties") && sch["properties"].is_object()) {
            std::vector<std::string> req;
            if (sch.contains("required") && sch["required"].is_array())
                for (const auto& r : sch["required"]) if (r.is_string()) req.push_back(r.get<std::string>());
            bool first = true;
            for (auto it = sch["properties"].begin(); it != sch["properties"].end(); ++it) {
                if (!first) o += ", ";
                first = false;
                o += it.key();
                if (std::find(req.begin(), req.end(), it.key()) == req.end()) o += "?";
                o += ": "; o += harmony_schema_to_ts(it.value());
            }
        }
        o += " }";
        return o;
    }
    return "any";
}

// Render the OpenAI `tools` array as a Harmony `namespace functions { … }` block
// (appended to the DEVELOPER message). This is the form gpt-oss is trained to read.
static void append_harmony_tools(std::string& dev, std::string_view tools_json) {
    nlohmann::ordered_json tools = nlohmann::ordered_json::parse(
        tools_json, nullptr, /*allow_exceptions=*/false);
    if (!tools.is_array() || tools.empty()) return;
    dev += "# Tools\n\n## functions\n\nnamespace functions {\n\n";
    for (const auto& t : tools) {
        const nlohmann::ordered_json* fn = nullptr;
        if (t.is_object() && t.contains("function")) fn = &t["function"];
        else if (t.is_object() && t.contains("name")) fn = &t;
        if (!fn || !fn->contains("name") || !(*fn)["name"].is_string()) continue;
        if (fn->contains("description") && (*fn)["description"].is_string() &&
            !(*fn)["description"].get<std::string>().empty()) {
            dev += "// "; dev += (*fn)["description"].get<std::string>(); dev += "\n";
        }
        std::string sig = "() => any;";
        if (fn->contains("parameters") && (*fn)["parameters"].is_object()) {
            const auto& p = (*fn)["parameters"];
            if (p.contains("properties") && p["properties"].is_object() && !p["properties"].empty())
                sig = "(_: " + harmony_schema_to_ts(p) + ") => any;";
        }
        dev += "type "; dev += (*fn)["name"].get<std::string>(); dev += " = "; dev += sig; dev += "\n\n";
    }
    dev += "} // namespace functions\n";
}

std::string build_harmony_prompt(std::span<const ChatTurn> turns,
                                 bool add_generation_prompt,
                                 std::string_view reasoning_effort,
                                 std::string_view tools_json) {
    // Current date for the canned system message (matches jinja strftime_now("%Y-%m-%d")).
    char date[16] = "1970-01-01";
    std::time_t now = std::time(nullptr);
    std::tm tmv{};
#if defined(_WIN32)
    localtime_s(&tmv, &now);
#else
    localtime_r(&now, &tmv);
#endif
    std::strftime(date, sizeof(date), "%Y-%m-%d", &tmv);

    std::string s;
    // Harmony SYSTEM message (always present).
    s += "<|start|>system<|message|>";
    s += "You are ChatGPT, a large language model trained by OpenAI.\n";
    s += "Knowledge cutoff: 2024-06\n";
    s += "Current date: "; s += date; s += "\n\n";
    s += "Reasoning: "; s += reasoning_effort; s += "\n\n";
    s += "# Valid channels: analysis, commentary, final. "
         "Channel must be included for every message.";
    if (!tools_json.empty())
        s += "\nCalls to these tools must go to the commentary channel.";
    s += "<|end|>";

    // DEVELOPER message = caller instructions (a leading "system" turn) + the
    // tool namespace. Emitted when either is present.
    size_t i = 0;
    const size_t n = turns.size();
    std::string dev;
    if (n > 0 && turns[0].role == "system") {
        if (!turns[0].content.empty()) {
            dev += "# Instructions\n\n"; dev += turns[0].content; dev += "\n\n";
        }
        i = 1;
    }
    if (!tools_json.empty()) append_harmony_tools(dev, tools_json);
    if (!dev.empty()) { s += "<|start|>developer<|message|>"; s += dev; s += "<|end|>"; }

    // Remaining turns. Tracks the last called function so its tool RESULT can be
    // addressed back to the assistant on the commentary channel.
    std::string last_fn;
    for (; i < n; ++i) {
        const auto& t = turns[i];
        if (t.role == "user") {
            s += "<|start|>user<|message|>"; s += t.content; s += "<|end|>";
        } else if (t.role == "assistant") {
            // An assistant turn carrying <tool_call> blocks is a prior tool call:
            // re-render each on the commentary channel (closing with <|call|>).
            bool any = false;
            size_t p = 0;
            for (;;) {
                size_t a = t.content.find("<tool_call>", p);
                if (a == std::string::npos) break;
                size_t b = t.content.find("</tool_call>", a);
                if (b == std::string::npos) break;
                const std::string inner = t.content.substr(a + 11, b - (a + 11));
                nlohmann::ordered_json call = nlohmann::ordered_json::parse(
                    inner, nullptr, /*allow_exceptions=*/false);
                if (call.is_object() && call.contains("name") && call["name"].is_string()) {
                    last_fn = call["name"].get<std::string>();
                    std::string args = "{}";
                    if (call.contains("arguments") && !call["arguments"].is_null())
                        args = call["arguments"].is_string()
                                 ? call["arguments"].get<std::string>()
                                 : call["arguments"].dump();
                    s += "<|start|>assistant<|channel|>commentary to=functions.";
                    s += last_fn; s += " <|constrain|>json<|message|>"; s += args; s += "<|call|>";
                    any = true;
                }
                p = b + 12;
            }
            if (!any) {
                s += "<|start|>assistant<|channel|>final<|message|>";
                s += t.content; s += "<|end|>";
            }
        } else if (t.role == "tool") {
            // Tool result → commentary channel, addressed back to the assistant.
            s += "<|start|>functions.";
            s += (last_fn.empty() ? std::string("tool") : last_fn);
            s += " to=assistant<|channel|>commentary<|message|>";
            s += t.content; s += "<|end|>";
        }
    }
    if (add_generation_prompt) s += "<|start|>assistant";
    return s;
}

}  // namespace ie
