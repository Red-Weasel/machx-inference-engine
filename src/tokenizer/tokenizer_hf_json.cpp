// src/tokenizer/tokenizer_hf_json.cpp — Tokenizer::load_from_hf_json: the HF `tokenizers` JSON source
// (DeepSeek-V4.1-Flash, docs/deepseek41/31). Its own translation unit: the unit tests compile
// tokenizer.cpp directly and need neither the JSON header nor this loader.
#include "ie/tokenizer.hpp"

#include "../../third_party/nlohmann/json.hpp"

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

namespace ie {

std::string Tokenizer::load_from_hf_json(const std::string& tokenizer_json, const std::string& tokenizer_config_json) {
    nlohmann::json tj, cj;
    { std::ifstream f(tokenizer_json); if (!f) return "cannot open " + tokenizer_json; try { f >> tj; } catch (const std::exception& e) { return std::string("tokenizer.json: ") + e.what(); } }
    { std::ifstream f(tokenizer_config_json); if (f) { try { f >> cj; } catch (const std::exception& e) { return std::string("tokenizer_config.json: ") + e.what(); } } }
    if (!tj.contains("model") || tj["model"].value("type", "") != "BPE") return "tokenizer.json: model.type is not BPE";
    const auto& model = tj["model"];
    vocab_.clear(); vocab_lookup_.clear(); merge_rank_.clear(); special_ids_.clear(); special_text_.clear(); scores_.clear();
    build_byte_maps();
    // the vocab: token -> id, plus the added tokens (ids past the base vocab; specials among them)
    const auto& vocab = model["vocab"];
    size_t n = 0;
    for (auto it = vocab.begin(); it != vocab.end(); ++it) n = std::max(n, size_t(it.value().get<int64_t>()) + 1);
    std::vector<std::pair<std::string, bool>> added;   // (content, special)
    if (tj.contains("added_tokens"))
        for (const auto& a : tj["added_tokens"]) { n = std::max(n, size_t(a["id"].get<int64_t>()) + 1); }
    vocab_.assign(n, std::string());
    for (auto it = vocab.begin(); it != vocab.end(); ++it) vocab_[size_t(it.value().get<int64_t>())] = it.key();
    if (tj.contains("added_tokens"))
        for (const auto& a : tj["added_tokens"]) {
            const int32_t id = a["id"].get<int32_t>(); const std::string content = a["content"].get<std::string>();
            vocab_[size_t(id)] = content;
            // every added token is matched as one literal (HF `tokenizers` does the same for added
            // tokens whether or not they are flagged special); `special` decides skip_special only
            special_text_.emplace_back(content, id);
            if (a.value("special", false)) special_ids_.insert(id);
        }
    for (size_t i = 0; i < vocab_.size(); ++i) if (!vocab_[i].empty()) vocab_lookup_.emplace(std::string_view(vocab_[i]), int32_t(i));
    std::sort(special_text_.begin(), special_text_.end(), [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    // the merges: "a b" strings (tokenizers 0.19 and earlier) or [a, b] pairs (later)
    const auto& merges = model["merges"];
    merge_rank_.reserve(merges.size());
    int32_t rank = 0;
    for (const auto& m : merges) {
        std::string a_str, b_str;
        if (m.is_string()) { const std::string ms = m.get<std::string>(); const auto sp = ms.find(' '); if (sp == std::string::npos) { ++rank; continue; } a_str = ms.substr(0, sp); b_str = ms.substr(sp + 1); }
        else if (m.is_array() && m.size() == 2) { a_str = m[0].get<std::string>(); b_str = m[1].get<std::string>(); }
        else { ++rank; continue; }
        const auto a_it = vocab_lookup_.find(a_str), b_it = vocab_lookup_.find(b_str);
        if (a_it != vocab_lookup_.end() && b_it != vocab_lookup_.end())
            merge_rank_.emplace((uint64_t(uint32_t(a_it->second)) << 32) | uint64_t(uint32_t(b_it->second)), rank);
        ++rank;
    }
    // bos / eos / pad from the config's token strings; add_bos_token as configured (V4.1: false)
    auto tok_id_of = [&](const char* key) -> int32_t {
        if (!cj.contains(key) || cj[key].is_null()) return -1;
        const std::string t = cj[key].is_object() ? cj[key].value("content", "") : cj[key].get<std::string>();
        const auto it = vocab_lookup_.find(t); return it == vocab_lookup_.end() ? -1 : it->second; };
    bos_id_ = tok_id_of("bos_token"); eos_id_ = tok_id_of("eos_token"); pad_id_ = tok_id_of("pad_token");
    add_bos_token_ = cj.value("add_bos_token", false);
    // the JSON's third Split is ` ?[\p{P}\p{S}]+...` with Unicode classes, so U+007E TILDE (Sm) is a
    // symbol: the cascade's tilde-as-symbol variant (the GGUF "joyai-llm" path classed it otherwise)
    pre_ = "joyai-llm"; joyai_ = true; hyv4_ = true; tekken_ = false; digits_1to3_ = false; ignore_merges_ = false; gemma_ = false; spm_ = false;
    pre_warning_.clear();
    return {};
}

}  // namespace ie
