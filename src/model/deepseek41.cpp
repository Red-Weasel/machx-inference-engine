// src/model/deepseek41.cpp — DeepSeek-V4.1-Flash config parse + weight binding (Phase 3).
// Host-only: no device memory, no compute. See include/ie/deepseek41.hpp.
#include "ie/deepseek41.hpp"

#include "../../third_party/nlohmann/json.hpp"

#include <algorithm>
#include <fstream>
#include <unordered_set>

namespace ie {

using json = nlohmann::json;

namespace {

// Every getter errors rather than defaulting: a config key this engine needs but the file does
// not carry is a bug to surface, not a value to invent.
struct Cfg {
    const json& j;
    std::string err;

    const json* at(const char* k) {
        auto it = j.find(k);
        if (it == j.end()) { if (err.empty()) err = std::string("config: missing key '") + k + "'"; return nullptr; }
        return &*it;
    }
    template <class T> void num(const char* k, T& dst) {
        if (const json* v = at(k)) {
            if (!v->is_number()) { if (err.empty()) err = std::string("config: '") + k + "' is not a number"; return; }
            dst = v->get<T>();
        }
    }
    void bl(const char* k, bool& dst) {
        if (const json* v = at(k)) {
            if (!v->is_boolean()) { if (err.empty()) err = std::string("config: '") + k + "' is not a bool"; return; }
            dst = v->get<bool>();
        }
    }
    void str(const char* k, std::string& dst) {
        if (const json* v = at(k)) {
            if (!v->is_string()) { if (err.empty()) err = std::string("config: '") + k + "' is not a string"; return; }
            dst = v->get<std::string>();
        }
    }
    template <class T> void arr(const char* k, std::vector<T>& dst) {
        if (const json* v = at(k)) {
            if (!v->is_array()) { if (err.empty()) err = std::string("config: '") + k + "' is not an array"; return; }
            dst.clear();
            for (const auto& e : *v) dst.push_back(e.get<T>());
        }
    }
};

// The scale sibling replaces the trailing ".weight", so `attn.wq_a.weight` pairs with
// `attn.wq_a.scale`, not `attn.wq_a.weight.scale`.
std::string scale_name(const std::string& weight) {
    static const std::string suf = ".weight";
    if (weight.size() > suf.size() && weight.compare(weight.size() - suf.size(), suf.size(), suf) == 0)
        return weight.substr(0, weight.size() - suf.size()) + ".scale";
    return weight + ".scale";
}

std::string shape_str(const std::vector<int64_t>& s) {
    std::string o = "[";
    for (size_t i = 0; i < s.size(); ++i) o += (i ? ", " : "") + std::to_string(s[i]);
    return o + "]";
}

// Binds by name, validates shape, records the claim. Any failure latches into `err` and every
// later call short-circuits, so the first real problem is the one reported.
class Binder {
public:
    Binder(const SafetensorsModel& m, std::unordered_set<std::string>& claimed)
        : m_(m), claimed_(claimed) {}

    // `scaled`: also bind `<name>.scale` and check its grid tiles `shape` by block_n x block_k.
    // block_n == 0 means "no scale plane" (BF16/F32 weights).
    void bind(Ds41Tensor& dst, const std::string& name, std::vector<int64_t> shape,
              uint32_t block_n = 0, uint32_t block_k = 0) {
        if (!err.empty()) return;
        const auto* t = m_.find(name);
        if (!t) { err = "missing tensor '" + name + "'"; return; }
        if (t->shape != shape) {
            err = "tensor '" + name + "' shape " + shape_str(t->shape) + ", expected " + shape_str(shape);
            return;
        }
        claimed_.insert(name);
        dst.w = t;
        if (!block_n) return;

        const std::string sn = scale_name(name);
        const auto* s = m_.find(sn);
        if (!s) { err = "missing scale plane '" + sn + "'"; return; }
        const std::vector<int64_t> want{(shape[0] + block_n - 1) / block_n,
                                        (shape[1] + block_k - 1) / block_k};
        if (s->shape != want) {
            err = "scale '" + sn + "' shape " + shape_str(s->shape) + ", expected " + shape_str(want) +
                  " (a " + std::to_string(block_n) + "x" + std::to_string(block_k) + " grid over " +
                  shape_str(shape) + ")";
            return;
        }
        claimed_.insert(sn);
        dst.s = s;
    }

    // An FP4 expert plane: stored [N, K/2] as I8, logical K, with a [N, K/32] E8M0 scale.
    void bind_fp4(Ds41Tensor& dst, const std::string& name, int64_t N, int64_t K) {
        bind(dst, name, {N, K / 2});        // shape only; the scale grid is 1x32 on K, below
        if (!err.empty()) return;
        const std::string sn = scale_name(name);
        const auto* s = m_.find(sn);
        if (!s) { err = "missing scale plane '" + sn + "'"; return; }
        const std::vector<int64_t> want{N, K / 32};
        if (s->shape != want) {
            err = "scale '" + sn + "' shape " + shape_str(s->shape) + ", expected " + shape_str(want);
            return;
        }
        claimed_.insert(sn);
        dst.s = s;
    }

    // A role the kind says must NOT be present. Binding nothing is the point; a tensor that
    // exists anyway means the kind is wrong, which is worth failing on.
    void forbid(const std::string& name) {
        if (err.empty() && m_.find(name)) err = "tensor '" + name + "' exists but this layer's kind forbids it";
    }

    std::string err;

private:
    const SafetensorsModel& m_;
    std::unordered_set<std::string>& claimed_;
};

}  // namespace

std::string ds41_parse_config(const std::string& json_text, Ds41Config& o) {
    json root;
    try { root = json::parse(json_text); }
    catch (const std::exception& e) { return std::string("config.json parse error: ") + e.what(); }

    if (!root.contains("model_type") || root["model_type"].get<std::string>() != "deepseek_v41")
        return "config.json is not model_type deepseek_v41";
    if (!root.contains("text_config")) return "config.json has no text_config";

    Cfg t{root["text_config"], {}};
    t.num("vocab_size", o.vocab_size);
    t.num("hidden_size", o.dim);
    t.num("moe_intermediate_size", o.moe_inter_dim);
    t.num("num_hidden_layers", o.n_layers);
    t.num("num_nextn_predict_layers", o.n_mtp_layers);
    t.num("num_attention_heads", o.n_heads);
    t.num("n_routed_experts", o.n_routed_experts);
    t.num("n_shared_experts", o.n_shared_experts);
    t.num("num_experts_per_tok", o.n_activated_experts);
    t.str("scoring_func", o.score_func);
    t.bl ("norm_topk_prob", o.norm_topk_prob);
    t.num("routed_scaling_factor", o.route_scale);
    t.num("swiglu_limit", o.swiglu_limit);
    t.num("q_lora_rank", o.q_lora_rank);
    t.num("head_dim", o.head_dim);
    t.num("qk_rope_head_dim", o.rope_head_dim);
    t.num("o_groups", o.o_groups);
    t.num("o_lora_rank", o.o_lora_rank);
    t.num("sliding_window", o.window_size);
    t.num("rms_norm_eps", o.norm_eps);
    t.arr("compress_ratios", o.compress_ratios);
    t.arr("kv_source_layer_ids", o.kv_source_layers);
    t.arr("index_source_layer_ids", o.index_source_layers);
    t.num("rope_theta", o.rope_theta);
    t.num("compress_rope_theta", o.compress_rope_theta);
    t.num("index_n_heads", o.index_n_heads);
    t.num("index_head_dim", o.index_head_dim);
    t.num("index_topk", o.index_topk);
    t.num("candidate_source_layer_id", o.candidate_source_layer);
    t.num("candidate_topk_blocks", o.candidate_topk_blocks);
    t.num("candidate_block_size", o.candidate_block_size);
    t.num("hc_mult", o.hc_mult);
    t.num("hc_sinkhorn_iters", o.hc_sinkhorn_iters);
    t.num("hc_eps", o.hc_eps);
    t.arr("engram_layer_ids", o.engram_layer_ids);
    t.arr("engram_num_embeddings", o.engram_num_embeddings);
    t.num("engram_max_ngram_size", o.engram_max_ngram_size);
    t.num("engram_n_heads", o.engram_n_heads);
    t.num("engram_head_dim", o.engram_head_dim);
    t.num("engram_vocab_size", o.engram_vocab_size);
    t.num("engram_compressed_vocab_size", o.engram_compressed_vocab_size);
    t.num("engram_pad_token_id", o.engram_pad_id);
    t.num("dspark_block_size", o.dspark_block_size);
    t.num("dspark_markov_rank", o.dspark_markov_rank);
    t.num("dspark_noise_token_id", o.dspark_noise_token_id);
    t.num("dspark_n_routed_experts", o.dspark_n_routed_experts);
    t.num("dspark_num_experts_per_tok", o.dspark_n_activated_experts);
    t.arr("dspark_target_layer_ids", o.dspark_target_layer_ids);
    if (!t.err.empty()) return t.err;

    // YaRN lives one level down and only exists when rope scaling is configured.
    if (const auto rs = t.j.find("rope_scaling"); rs != t.j.end() && rs->is_object()) {
        Cfg r{*rs, {}};
        r.num("factor", o.rope_factor);
        r.num("beta_fast", o.beta_fast);
        r.num("beta_slow", o.beta_slow);
        r.num("original_max_position_embeddings", o.original_seq_len);
        if (!r.err.empty()) return r.err;
    }
    if (root.contains("image_token_id")) o.image_token_id = root["image_token_id"].get<uint32_t>();
    if (const auto v = root.find("vision_config"); v != root.end() && v->is_object()) {
        Cfg vc{*v, {}};
        vc.num("num_hidden_layers", o.vision_n_layers);
        vc.num("hidden_size", o.vision_dim);
        if (!vc.err.empty()) return vc.err;
    }

    const uint32_t n_total = o.n_layers + o.n_mtp_layers;
    if (o.compress_ratios.size() < n_total)
        return "config: compress_ratios has " + std::to_string(o.compress_ratios.size()) +
               " entries, need " + std::to_string(n_total);
    if (o.hc_mult == 0 || o.head_dim <= o.rope_head_dim || o.o_groups == 0)
        return "config: implausible shapes (hc_mult/head_dim/o_groups)";
    return {};
}

std::string DeepSeek41Model::load(const std::string& dir) {
    {
        std::ifstream f(dir + "/config.json", std::ios::binary);
        if (!f) return "cannot open " + dir + "/config.json";
        const std::string body((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (auto e = ds41_parse_config(body, cfg_); !e.empty()) return e;
    }
    if (auto e = store_.open(dir); !e.empty()) return e;

    const auto& c = cfg_;
    const int64_t H = c.dim, HD = c.head_dim, QR = c.q_lora_rank;
    const int64_t QH = int64_t(c.n_heads) * HD;                 // 64 * 512 = 32768
    const int64_t OR = int64_t(c.o_groups) * c.o_lora_rank;     //  8 * 1024 =  8192
    const int64_t IHD = int64_t(c.index_n_heads) * c.index_head_dim;
    const int64_t MIX = int64_t(2 + c.hc_mult) * c.hc_mult;     // (2+4)*4 = 24
    const uint32_t n_total = c.n_layers + c.n_mtp_layers;

    std::unordered_set<std::string> claimed;
    claimed.reserve(size_t(store_.all().size()) * 2);
    Binder b(store_, claimed);

    b.bind(embed,      "embed.weight", {int64_t(c.vocab_size), H});
    b.bind(lm_head,    "head.weight",  {int64_t(c.vocab_size), H});
    b.bind(final_norm, "norm.weight",  {H});
    if (c.dspark_block_size && c.n_mtp_layers) {   // DSpark P1 (docs/57): the drafter's extras
        const std::string last = "mtp." + std::to_string(c.n_mtp_layers - 1) + ".";
        b.bind(dspark.main_proj, "mtp.0.main_proj.weight", {H, H * int64_t(c.dspark_target_layer_ids.size())}, 32, 32);
        b.bind(dspark.main_norm, "mtp.0.main_norm.weight", {H});
        b.bind(dspark.norm, last + "norm.weight", {H});
        b.bind(dspark.markov_embed, last + "markov_head.embed.weight", {int64_t(c.vocab_size), int64_t(c.dspark_markov_rank)});
        b.bind(dspark.markov_head, last + "markov_head.head.weight", {int64_t(c.vocab_size), int64_t(c.dspark_markov_rank)});
        b.bind(dspark.confidence_proj, last + "confidence_head.proj.weight", {1, H + int64_t(c.dspark_markov_rank)});
    }
    if (!b.err.empty()) return b.err;

    layers_.assign(n_total, {});
    for (uint32_t L = 0; L < n_total; ++L) {
        auto& w = layers_[L];
        const bool backbone = L < c.n_layers;
        const std::string p = backbone ? "layers." + std::to_string(L) + "."
                                       : "mtp." + std::to_string(L - c.n_layers) + ".";

        auto& k = w.kind;
        k.compress_ratio = c.compress_ratios[L];
        // Only backbone layers can be sources: the reference gates both on `is_backbone`.
        k.is_kv_source = backbone && std::find(c.kv_source_layers.begin(), c.kv_source_layers.end(), L)
                                         != c.kv_source_layers.end();
        k.is_index_source = backbone && std::find(c.index_source_layers.begin(),
                                                  c.index_source_layers.end(), L)
                                            != c.index_source_layers.end();
        // Compressor.__init__ builds a gate only when it pools, i.e. compress_ratio > 1.
        k.has_compressor_gate = k.is_kv_source && k.compress_ratio > 1;
        k.has_engram = backbone && std::find(c.engram_layer_ids.begin(), c.engram_layer_ids.end(), L)
                                       != c.engram_layer_ids.end();
        k.n_routed    = c.n_routed(L);
        k.n_activated = c.n_activated(L);

        b.bind(w.attn_norm, p + "attn_norm.weight", {H});
        b.bind(w.wq_a,      p + "attn.wq_a.weight", {QR, H},  32, 32);
        b.bind(w.q_norm,    p + "attn.q_norm.weight", {QR});
        b.bind(w.wq_b,      p + "attn.wq_b.weight", {QH, QR}, 32, 32);
        b.bind(w.wkv,       p + "attn.wkv.weight",  {HD, H},  32, 32);
        b.bind(w.kv_norm,   p + "attn.kv_norm.weight", {HD});
        b.bind(w.attn_sink, p + "attn.attn_sink", {int64_t(c.n_heads)});
        b.bind(w.wo_a,      p + "attn.wo_a.weight", {OR, QH / int64_t(c.o_groups)}, 32, 32);
        b.bind(w.wo_b,      p + "attn.wo_b.weight", {H, OR}, 32, 32);

        if (k.is_kv_source) {
            b.bind(w.comp_wkv,  p + "attn.compressor.wkv.weight",  {HD, H});
            b.bind(w.comp_norm, p + "attn.compressor.norm.weight", {HD});
            if (k.has_compressor_gate) b.bind(w.comp_wgate, p + "attn.compressor.wgate.weight", {HD, H});
            else                       b.forbid(p + "attn.compressor.wgate.weight");
        } else {
            b.forbid(p + "attn.compressor.wkv.weight");
        }

        if (k.is_index_source) {
            b.bind(w.idx_wq_b,    p + "attn.indexer.wq_b.weight", {IHD, QR}, 32, 32);
            b.bind(w.idx_weights, p + "attn.indexer.weights_proj.weight", {int64_t(c.index_n_heads), H});
            // The indexer's K comes from the shared compressed latent, so it exists exactly
            // where that latent is produced -- on the KV-source layers, not every index source.
            if (k.is_kv_source) {
                b.bind(w.idx_wk,     p + "attn.indexer.wk.weight", {int64_t(c.index_head_dim), HD});
                b.bind(w.idx_k_norm, p + "attn.indexer.k_norm.weight", {int64_t(c.index_head_dim)});
            } else {
                b.forbid(p + "attn.indexer.wk.weight");
            }
        } else {
            b.forbid(p + "attn.indexer.wq_b.weight");
        }

        b.bind(w.ffn_norm,     p + "ffn_norm.weight", {H});
        b.bind(w.gate_w,       p + "ffn.gate.weight", {int64_t(k.n_routed), H});
        b.bind(w.gate_bias,    p + "ffn.gate.bias",    {int64_t(k.n_routed)});
        b.bind(w.gate_bias_vl, p + "ffn.gate.bias_vl", {int64_t(k.n_routed)});

        const int64_t FI = c.moe_inter_dim;
        w.exp_w1.resize(k.n_routed); w.exp_w3.resize(k.n_routed); w.exp_w2.resize(k.n_routed);
        for (uint32_t E = 0; E < k.n_routed && b.err.empty(); ++E) {
            const std::string ep = p + "ffn.experts." + std::to_string(E) + ".";
            b.bind_fp4(w.exp_w1[E], ep + "w1.weight", FI, H);
            b.bind_fp4(w.exp_w3[E], ep + "w3.weight", FI, H);
            b.bind_fp4(w.exp_w2[E], ep + "w2.weight", H,  FI);
        }
        b.bind(w.sh_w1, p + "ffn.shared_experts.w1.weight", {FI, H}, 32, 32);
        b.bind(w.sh_w3, p + "ffn.shared_experts.w3.weight", {FI, H}, 32, 32);
        b.bind(w.sh_w2, p + "ffn.shared_experts.w2.weight", {H, FI}, 32, 32);

        const int64_t HCD = int64_t(c.hc_mult) * H;
        b.bind(w.hc_attn_fn,    p + "hc_attn_fn",    {MIX, HCD});
        b.bind(w.hc_attn_base,  p + "hc_attn_base",  {MIX});
        b.bind(w.hc_attn_scale, p + "hc_attn_scale", {3});
        b.bind(w.hc_ffn_fn,     p + "hc_ffn_fn",     {MIX, HCD});
        b.bind(w.hc_ffn_base,   p + "hc_ffn_base",   {MIX});
        b.bind(w.hc_ffn_scale,  p + "hc_ffn_scale",  {3});

        if (k.has_engram) {
            const auto it = std::find(c.engram_layer_ids.begin(), c.engram_layer_ids.end(), L);
            const size_t ei = size_t(it - c.engram_layer_ids.begin());
            if (ei >= c.engram_num_embeddings.size())
                return "config: engram_num_embeddings has no entry for layer " + std::to_string(L);
            const int64_t rows = int64_t(c.engram_num_embeddings[ei]);
            // The embedding table's scale is one E8M0 per 32 along the row: [rows, head_dim/32].
            b.bind(w.engram_embed, p + "engram.embed.weight",
                   {rows, int64_t(c.engram_head_dim)}, 1, 32);
            b.bind(w.engram_q, p + "engram.q_weight", {int64_t(c.engram_max_ngram_size), H});
            b.bind(w.engram_k, p + "engram.k_weight", {int64_t(c.engram_max_ngram_size), H});
            // wkv shape is not derivable from the documented fields alone; take the file's.
            if (b.err.empty()) {
                const auto* t = store_.find(p + "engram.wkv.weight");
                if (!t) { b.err = "missing tensor '" + p + "engram.wkv.weight'"; }
                else {
                    w.engram_wkv.w = t; claimed.insert(t->name);
                    if (const auto* s = store_.find(p + "engram.wkv.scale")) {
                        w.engram_wkv.s = s; claimed.insert(s->name);
                    }
                }
            }
        } else {
            b.forbid(p + "engram.embed.weight");
        }
        if (!b.err.empty()) return "layer " + std::to_string(L) + ": " + b.err;
    }

    // Roles the MTP blocks carry beyond a backbone Block, and the vision tower: bound by name
    // rather than by role, because nothing in this phase consumes them. They still have to be
    // claimed, or criterion 3(c) would report them as unknown.
    for (const auto* t : store_.all()) {
        if (claimed.count(t->name)) continue;
        const auto& n = t->name;
        const bool vision = n.rfind("vision.", 0) == 0 || n.rfind("aligner.", 0) == 0 ||
                            n == "image_start" || n == "image_end" || n == "image_newline";
        if (vision) { claimed.insert(n); continue; }
        unclaimed_.push_back(n);
    }
    std::sort(unclaimed_.begin(), unclaimed_.end());

    // Byte accounting, from the same tensor table the binder used.
    for (const auto* t : store_.all()) {
        const auto& n = t->name;
        uint64_t* bucket = &budget_.norms_gates_hc;
        if (n.rfind("vision.", 0) == 0 || n.rfind("aligner.", 0) == 0 ||
            n == "image_start" || n == "image_end" || n == "image_newline") bucket = &budget_.vision;
        else if (n.rfind("mtp.", 0) == 0)                       bucket = &budget_.mtp;
        else if (n.find(".engram.") != std::string::npos)       bucket = &budget_.engram;
        else if (n.find(".ffn.experts.") != std::string::npos)  bucket = &budget_.routed_experts;
        else if (n.find(".ffn.shared_experts.") != std::string::npos) bucket = &budget_.shared_experts;
        else if (n.find(".attn.indexer.") != std::string::npos) bucket = &budget_.indexer;
        else if (n.find(".attn.compressor.") != std::string::npos) bucket = &budget_.compressor;
        else if (n.find(".attn.") != std::string::npos)         bucket = &budget_.attention;
        else if (n == "embed.weight")                           bucket = &budget_.embed;
        else if (n == "head.weight")                            bucket = &budget_.lm_head;
        *bucket += t->nbytes;
    }
    return {};
}

}  // namespace ie
