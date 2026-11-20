// src/model/hy4flash.cpp — Hy4-flash config parse + weight binding (docs/hy4flash/00_PORT_AND_ABLITERATION_PLAN.md).
// Host-only: no device memory, no compute. See include/ie/hy4flash.hpp.
#include "ie/hy4flash.hpp"
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include "../../third_party/nlohmann/json.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <unordered_set>

namespace ie {

using json = nlohmann::json;

namespace {

std::string shape_str(const std::vector<int64_t>& s) {
    std::string o = "[";
    for (size_t i = 0; i < s.size(); ++i) o += (i ? ", " : "") + std::to_string(s[i]);
    return o + "]";
}

std::string read_file(const std::string& path, std::string& body) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "cannot open " + path;
    body.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return {};
}

// Every getter errors rather than defaulting (the V4.1 binder's discipline): a key this engine needs and the file does
// not carry is a bug to surface, not a value to invent.
struct Cfg {
    const json& j;
    std::string err;
    const json* at(const char* k) {
        auto it = j.find(k);
        if (it == j.end() || it->is_null()) { if (err.empty()) err = std::string("config: missing key '") + k + "'"; return nullptr; }
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
};

bool is_float_dtype(const std::string& d) { return d == "BF16" || d == "F32" || d == "F16"; }

// Binds by name, validates the shape and the dtype class, records the claim. The first failure latches into `err`.
class Binder {
public:
    Binder(const SafetensorsModel& m, std::unordered_set<std::string>& claimed, const SafetensorsReader* overlay = nullptr)
        : m_(m), claimed_(claimed), ovl_(overlay) {}
    size_t overlaid = 0;
    void fl(Ds41Tensor& dst, const std::string& name, std::vector<int64_t> shape) { dst.w = one(name, shape, 0); }
    void i32(Ds41Tensor& dst, const std::string& name, std::vector<int64_t> shape) { dst.w = one(name, shape, 1); }
    void u8(Ds41Tensor& dst, const std::string& name, std::vector<int64_t> shape, const std::string& scale_name, std::vector<int64_t> scale_shape) {
        dst.w = one(name, shape, 2); dst.s = one(scale_name, scale_shape, 2);
    }
    const SafeTensorInfo* peek(const std::string& name) const { return m_.find(name); }
    std::string err;
private:
    const SafeTensorInfo* one(const std::string& name, const std::vector<int64_t>& shape, int kind) {
        if (!err.empty()) return nullptr;
        const SafeTensorInfo* t = ovl_ ? ovl_->find(name) : nullptr;      // an overlay tensor stands in for the source's
        if (t) { if (!m_.find(name)) { err = "overlay tensor '" + name + "' has no counterpart in the source"; return nullptr; } ++overlaid; }
        else t = m_.find(name);
        if (!t) { err = "missing tensor '" + name + "'"; return nullptr; }
        const bool ok = kind == 0 ? is_float_dtype(t->dtype_str) : kind == 1 ? t->dtype_str == "I32" : t->dtype_str == "U8";
        if (!ok || t->shape != shape) {
            err = "tensor '" + name + "' is " + t->dtype_str + " " + shape_str(t->shape) + ", expected " +
                  (kind == 0 ? "a float type " : kind == 1 ? "I32 " : "U8 ") + shape_str(shape);
            return nullptr;
        }
        claimed_.insert(name);
        return t;
    }
    const SafetensorsModel& m_;
    std::unordered_set<std::string>& claimed_;
    const SafetensorsReader* ovl_;
};

bool file_exists(const std::string& p) { std::ifstream f(p); return bool(f); }

}  // namespace

std::string hy4f_parse_config(const std::string& json_text, Hy4fConfig& o, std::vector<std::vector<uint64_t>>* engram_head_sizes) {
    json top;
    try { top = json::parse(json_text); }
    catch (const std::exception& e) { return std::string("config.json parse error: ") + e.what(); }
    if (!top.contains("text_config") || !top["text_config"].is_object()) return "config: no text_config object (not a Hy4-flash config.json)";
    if (top.value("model_type", std::string()) != "hy_v4_flash") return "config: model_type is not hy_v4_flash";
    const json& j = top["text_config"];
    Cfg c{j, {}};
    c.num("vocab_size", o.vocab_size); c.num("hidden_size", o.dim); c.num("intermediate_size", o.inter_dim);
    c.num("moe_intermediate_size", o.moe_inter_dim); c.num("routed_expert_hidden_size", o.latent_dim);
    c.num("num_hidden_layers", o.n_layers);
    c.num("num_attention_heads", o.n_heads); c.num("swa_num_attention_heads", o.n_heads_swa);
    c.num("q_lora_rank", o.q_lora_rank); c.num("kv_lora_rank", o.kv_lora_rank);
    c.num("qk_nope_head_dim", o.nope_dim); c.num("qk_rope_head_dim", o.rope_dim); c.num("v_head_dim", o.v_dim);
    uint32_t sn = 0, sr = 0, sv = 0;
    c.num("swa_qk_nope_head_dim", sn); c.num("swa_qk_rope_head_dim", sr); c.num("swa_v_head_dim", sv);
    c.num("sliding_window", o.window);
    c.num("n_routed_experts", o.n_experts); c.num("num_experts_per_tok", o.top_k); c.num("n_shared_experts", o.n_shared);
    c.num("routed_scaling_factor", o.route_scale); c.num("swiglu_limit", o.swiglu_limit); c.bl("norm_topk_prob", o.norm_topk_prob);
    c.num("hc_mult", o.hc_mult); c.num("hc_magnitude", o.hc_magnitude); c.num("hc_eps", o.hc_eps); c.num("rms_norm_eps", o.norm_eps);
    c.num("compressed_indexer_compress_ratio", o.ix_ratio); c.num("compressed_indexer_num_attention_heads", o.ix_heads);
    c.num("compressed_indexer_head_dim", o.ix_head_dim); c.num("compressed_indexer_group_topk", o.ix_group_topk);
    c.num("engram_max_ngram_size", o.engram_max_ngram); c.num("engram_n_head_per_ngram", o.engram_heads);
    c.num("engram_n_embed_per_ngram", o.engram_embed); c.num("engram_pad_id", o.engram_pad_id);
    c.bl("engram_fp8_embedding", o.engram_fp8);
    c.num("bos_token_id", o.bos_id); c.num("eos_token_id", o.eos_id); c.num("pad_token_id", o.pad_id);
    uint32_t n_group = 0, topk_group = 0; bool attn_bias = true;
    c.num("n_group", n_group); c.num("topk_group", topk_group); c.bl("attention_bias", attn_bias);
    if (!c.err.empty()) return c.err;
    o.engram_seed = j.value("engram_seed", 0);
    o.image_token_id = top.value("image_token_id", -1); o.video_token_id = top.value("video_token_id", -1);
    o.image_start_id = top.value("image_start_token_id", -1); o.image_end_id = top.value("image_end_token_id", -1);
    o.dfly_layers = top.value("dfly_num_hidden_layers", 0u); o.dfly_mask_id = top.value("dfly_mask_token_id", -1);
    if (top.contains("dfly_target_kv_layers") && top["dfly_target_kv_layers"].is_array()) for (const auto& v : top["dfly_target_kv_layers"]) o.dfly_kv_layers.push_back(v.get<uint32_t>());
    if (top.contains("speculative_config") && top["speculative_config"].is_object()) o.dfly_drafts = top["speculative_config"].value("num_speculative_tokens", 0u);
    if (sn != o.nope_dim || sr != o.rope_dim || sv != o.v_dim) return "config: the sliding layers' head dims differ from the indexed layers' (not supported)";
    if (n_group != 1 || topk_group != 1) return "config: router groups (n_group / topk_group != 1) are not supported";
    if (attn_bias) return "config: attention_bias is not supported";
    if (o.n_shared != 1) return "config: n_shared_experts != 1 is not supported";
    if (j.value("hidden_act", std::string()) != "silu") return "config: hidden_act is not silu";
    if (o.hc_mult == 0 || o.hc_mult > 8) return "config: hc_mult out of range";
    if (o.rope_dim % 2 || o.latent_dim % 32 || o.moe_inter_dim % 32) return "config: odd rope width or an expert width that is not a multiple of 32";

    const std::string dt = top.value("dtype", top.value("torch_dtype", std::string("bfloat16")));
    if (dt == "bfloat16") o.torch_default_eps = 0.0078125f;
    else if (dt == "float32") o.torch_default_eps = 1.1920929e-07f;
    else if (dt == "float16") o.torch_default_eps = 0.0009765625f;
    else return "config: dtype '" + dt + "' is not one this engine knows the default norm epsilon of";

    auto is_indexed = [](const std::string& s, bool& ok) {
        if (s == "indexed_attention" || s == "deepseek_sparse_attention") return true;
        if (s != "sliding_attention") ok = false;
        return false;
    };
    if (!j.contains("layer_types") || !j["layer_types"].is_array() || j["layer_types"].size() != o.n_layers) return "config: layer_types missing or not one per layer";
    if (!j.contains("mlp_layer_types") || !j["mlp_layer_types"].is_array() || j["mlp_layer_types"].size() != o.n_layers) return "config: mlp_layer_types missing or not one per layer";
    o.indexed.clear(); o.moe.clear();
    for (uint32_t L = 0; L < o.n_layers; ++L) {
        bool ok = true;
        o.indexed.push_back(is_indexed(j["layer_types"][L].get<std::string>(), ok) ? 1 : 0);
        if (!ok) return "config: layer_types[" + std::to_string(L) + "] is not a known attention type";
        const std::string mt = j["mlp_layer_types"][L].get<std::string>();
        if (mt != "sparse" && mt != "dense") return "config: mlp_layer_types[" + std::to_string(L) + "] is neither sparse nor dense";
        o.moe.push_back(mt == "sparse" ? 1 : 0);
    }
    if (!j.contains("rope_parameters") || !j["rope_parameters"].is_object()) return "config: rope_parameters missing";
    bool got_i = false, got_s = false;
    for (auto it = j["rope_parameters"].begin(); it != j["rope_parameters"].end(); ++it) {
        bool ok = true; const bool ix = is_indexed(it.key(), ok);
        if (!ok) return "config: rope_parameters has an unknown layer type '" + it.key() + "'";
        const json& r = it.value();
        if (r.value("rope_type", std::string("default")) != "default") return "config: rope_type is not default";
        if (!r.contains("rope_theta") || !r.contains("mrope_section") || r["mrope_section"].size() != 3) return "config: rope_parameters['" + it.key() + "'] lacks rope_theta or a 3-entry mrope_section";
        (ix ? o.theta_indexed : o.theta_swa) = r["rope_theta"].get<float>();
        for (int a = 0; a < 3; ++a) (ix ? o.mrope_indexed : o.mrope_swa)[a] = r["mrope_section"][a].get<uint32_t>();
        (ix ? got_i : got_s) = true;
    }
    if (!got_i || !got_s) return "config: rope_parameters must name both attention types";

    o.engram_layer_ids.clear();
    if (engram_head_sizes) engram_head_sizes->clear();
    if (j.contains("engram_layer_ids") && j["engram_layer_ids"].is_array()) {
        if (!j.contains("engram_layer_head_vocab_sizes") || !j["engram_layer_head_vocab_sizes"].is_object()) return "config: engram_layer_head_vocab_sizes missing";
        for (const auto& idj : j["engram_layer_ids"]) {
            const uint32_t id = idj.get<uint32_t>();
            if (id == 0 || id > o.n_layers) return "config: engram layer id " + std::to_string(id) + " out of range";
            const auto it = j["engram_layer_head_vocab_sizes"].find(std::to_string(id));
            if (it == j["engram_layer_head_vocab_sizes"].end() || it->size() != o.engram_max_ngram - 1) return "config: engram_layer_head_vocab_sizes['" + std::to_string(id) + "'] missing or not one list per n-gram order";
            std::vector<uint64_t> sizes;
            for (const auto& order : *it) {
                if (order.size() != o.engram_heads) return "config: an n-gram order of layer " + std::to_string(id) + " does not list engram_n_head_per_ngram sizes";
                for (const auto& s : order) sizes.push_back(s.get<uint64_t>());
            }
            o.engram_layer_ids.push_back(id);
            if (engram_head_sizes) engram_head_sizes->push_back(std::move(sizes));
        }
        if (o.engram_embed % o.engram_heads) return "config: engram_n_embed_per_ngram is not a multiple of engram_n_head_per_ngram";
    }
    return {};
}

std::string hy4f_to_f32(const Ds41Tensor& t, std::vector<float>& out) {
    if (!t.w) return "to_f32: unbound tensor";
    const size_t n = size_t(t.w->numel());
    out.resize(n);
    const uint8_t* p = t.w->data;
    if (t.w->dtype_str == "F32") { std::memcpy(out.data(), p, n * 4); return {}; }
    if (t.w->dtype_str == "BF16") {
        for (size_t i = 0; i < n; ++i) { uint16_t b; std::memcpy(&b, p + i * 2, 2); const uint32_t u = uint32_t(b) << 16; std::memcpy(&out[i], &u, 4); }
        return {};
    }
    if (t.w->dtype_str == "F16") {
        for (size_t i = 0; i < n; ++i) {
            uint16_t h; std::memcpy(&h, p + i * 2, 2);
            const uint32_t s = uint32_t(h >> 15) << 31, e = (h >> 10) & 31u, m = h & 1023u;
            uint32_t u;
            if (e == 0) {
                if (m == 0) u = s;
                else { uint32_t mm = m; int sh = 0; while (!(mm & 1024u)) { mm <<= 1; ++sh; } u = s | (uint32_t(113 - sh) << 23) | ((mm & 1023u) << 13); }
            } else if (e == 31) u = s | 0x7F800000u | (m << 13);
            else u = s | ((e + 112u) << 23) | (m << 13);
            std::memcpy(&out[i], &u, 4);
        }
        return {};
    }
    return "to_f32: '" + t.w->name + "' is " + t.w->dtype_str;
}

void hy4f_engram_rows(const Hy4fEngram& eg, const Hy4fConfig& cfg, const int32_t* lookup, const int32_t* ngram, uint64_t* rows) {
    // HYV4FlashEngramLayer.hash_ngrams: token -> compressed id (tokens past the table read the pad id's), each multiplied
    // by its position's odd multiplier in int64, the order-n hash = xor of the first n products, then one modulus a head.
    int64_t mixed[8];
    for (uint32_t j = 0; j < cfg.engram_max_ngram; ++j) {
        const int32_t tok = ngram[j];
        const int32_t c = (tok >= 0 && uint32_t(tok) < eg.table_size) ? lookup[tok] : lookup[cfg.engram_pad_id];
        mixed[j] = int64_t(uint64_t(int64_t(c)) * uint64_t(eg.multipliers[j]));
    }
    for (uint32_t n = 2; n <= cfg.engram_max_ngram; ++n) {
        int64_t mix = mixed[0];
        for (uint32_t off = 1; off < n; ++off) mix ^= mixed[off];
        const uint32_t h0 = (n - 2) * cfg.engram_heads;
        for (uint32_t h = 0; h < cfg.engram_heads; ++h) {
            const int64_t mod = int64_t(eg.head_sizes[h0 + h]);
            int64_t r = mix % mod; if (r < 0) r += mod;                 // torch's %: the sign of the divisor
            rows[h0 + h] = uint64_t(r) + eg.head_offsets[h0 + h];
        }
    }
}

Hy4FlashModel::~Hy4FlashModel() { for (auto& m : maps_) ::munmap(m.first, m.second); }

std::string Hy4FlashModel::load(const std::string& dir) {
    dir_ = dir; has_src_experts_ = true;
    for (auto& m : maps_) ::munmap(m.first, m.second);
    maps_.clear();
    // ---- the sidecar and the config ----
    std::string body;
    if (auto e = read_file(dir + "/hy4flash_ie.json", body); !e.empty())
        return e + " (an engine directory written by scripts/hy4flash_oracle/pack.py is expected, not the released checkpoint itself)";
    json side;
    try { side = json::parse(body); } catch (const std::exception& e) { return std::string("hy4flash_ie.json parse error: ") + e.what(); }
    if (side.value("format", 0) != 1) return "hy4flash_ie.json: unknown format";
    source_dir_ = side.value("source_dir", std::string());
    if (source_dir_.empty()) return "hy4flash_ie.json: no source_dir";
    if (auto e = read_file(dir + "/config.json", body); !e.empty()) return e;
    std::vector<std::vector<uint64_t>> head_sizes;
    if (auto e = hy4f_parse_config(body, cfg_, &head_sizes); !e.empty()) return e;
    const Hy4fConfig& c = cfg_;

    if (auto e = src_.open(source_dir_); !e.empty())
        return "source directory " + source_dir_ + ": " + e + " (is its drive mounted?)";
    has_pack_ = false;
    const bool pack_listed = side.value("expert_format", std::string("none")) == "mxfp4";
    if (pack_listed && file_exists(dir + "/model.safetensors.index.json")) {
        if (auto e = pack_.open(dir); !e.empty()) return "expert pack: " + e;
        has_pack_ = true;
    }

    const bool has_overlay = file_exists(dir + "/overlay.safetensors");
    if (has_overlay) { if (auto e = ovl_.open(dir + "/overlay.safetensors"); !e.empty()) return "overlay.safetensors: " + e; }
    std::unordered_set<std::string> claimed, pack_claimed;
    Binder b(src_, claimed, has_overlay ? &ovl_ : nullptr), pb(pack_, pack_claimed);
    const int64_t H = c.dim, HC = c.hc_mult, V = c.vocab_size, E = c.n_experts, MI = c.moe_inter_dim, LAT = c.latent_dim;
    const int64_t QL = c.q_lora_rank, KL = c.kv_lora_rank, IXD = c.ix_head_dim;

    b.fl(embed, "model.embed_tokens.weight", {V, H});
    b.fl(lm_head, "lm_head.weight", {V, H});
    b.fl(final_norm, "model.norm.weight", {H});
    b.fl(head_fn, "model.hc_head.hc_head_fn", {HC, HC * H});
    b.fl(head_base, "model.hc_head.hc_head_base", {HC});
    b.fl(head_scale, "model.hc_head.hc_head_scale", {1});

    layers_.assign(c.n_layers, Hy4fLayer{});
    engrams_.clear();
    dense_f16_ = uint64_t(V) * H * 2;
    auto mat = [&](Ds41Tensor& dst, const std::string& name, int64_t N, int64_t K) { b.fl(dst, name, {N, K}); dense_f16_ += uint64_t(N) * K * 2; };
    bool all_packed = has_pack_;
    for (uint32_t L = 0; L < c.n_layers && b.err.empty() && pb.err.empty(); ++L) {
        Hy4fLayer& l = layers_[L];
        l.indexed = c.indexed[L]; l.moe = c.moe[L]; l.n_heads = c.heads(L);
        const std::string p = "model.layers." + std::to_string(L) + ".";
        const int64_t NH = l.n_heads;
        b.fl(l.hc_attn_fn, p + "hc_attn_layer.hc_pre.hc_fn", {2 * HC, HC * H});
        b.fl(l.hc_attn_base, p + "hc_attn_layer.hc_pre.hc_base", {2 * HC});
        b.fl(l.hc_attn_scale, p + "hc_attn_layer.hc_pre.hc_scale", {2});
        b.fl(l.hc_ffn_fn, p + "hc_mlp_layer.hc_pre.hc_fn", {2 * HC, HC * H});
        b.fl(l.hc_ffn_base, p + "hc_mlp_layer.hc_pre.hc_base", {2 * HC});
        b.fl(l.hc_ffn_scale, p + "hc_mlp_layer.hc_pre.hc_scale", {2});
        b.fl(l.attn_norm, p + "input_layernorm.weight", {H});
        b.fl(l.ffn_norm, p + "post_attention_layernorm.weight", {H});
        const std::string a = p + "self_attn.";
        mat(l.q_a, a + "q_a_proj.weight", QL, H);
        b.fl(l.q_a_norm, a + "q_a_layernorm.weight", {QL});
        mat(l.q_b, a + "q_b_proj.weight", NH * c.qk_dim(), QL);
        mat(l.kv_a, a + "kv_a_proj_with_mqa.weight", KL + c.rope_dim, H);
        b.fl(l.kv_a_norm, a + "kv_a_layernorm.weight", {KL});
        mat(l.kv_b, a + "kv_b_proj.weight", NH * (c.nope_dim + c.v_dim), KL);
        mat(l.gate, a + "linear_gate.weight", NH * c.v_dim, H);
        mat(l.o, a + "o_proj.weight", H, NH * c.v_dim);
        b.fl(l.post_norm, a + "post_proj_norm.weight", {H});
        b.fl(l.sinks, a + "learnable_sink_param", {NH});
        if (l.indexed) {
            const std::string x = a + "compressed_indexer.";
            mat(l.ix_q, x + "query_projection.weight", int64_t(c.ix_heads) * IXD, QL);
            b.fl(l.ix_head_w, x + "head_weight_projection.weight", {c.ix_heads, H});
            b.fl(l.ix_tok_k, x + "token_key_projection.weight", {IXD, H});
            b.fl(l.ix_tok_norm, x + "token_key_norm.weight", {IXD});
            b.fl(l.ix_kv, x + "key_value_projection.weight", {IXD, IXD});
            b.fl(l.ix_kg, x + "key_gate_projection.weight", {IXD, IXD});
            b.fl(l.ix_key_norm, x + "key_norm.weight", {IXD});
            b.fl(l.ix_ape, x + "intra_group_ape", {c.ix_ratio, IXD});
        }
        const std::string f = p + "mlp.";
        if (!l.moe) {
            mat(l.mlp_gate, f + "gate_proj.weight", c.inter_dim, H);
            mat(l.mlp_up, f + "up_proj.weight", c.inter_dim, H);
            mat(l.mlp_down, f + "down_proj.weight", H, c.inter_dim);
            b.fl(l.mlp_post_norm, f + "post_fc2_norm.weight", {H});
        } else {
            b.fl(l.router_w, f + "gate.weight", {E, H});
            b.fl(l.router_bias, f + "gate.e_score_correction_bias", {E});
            mat(l.lat_down, f + "routed_expert_down_proj.weight", LAT, H);
            mat(l.lat_up, f + "routed_expert_up_proj.weight", H, LAT);
            mat(l.sh_gate, f + "shared_experts.gate_proj.weight", MI, H);
            mat(l.sh_up, f + "shared_experts.up_proj.weight", MI, H);
            mat(l.sh_down, f + "shared_experts.down_proj.weight", H, MI);
            b.fl(l.moe_norm, f + "combine_output_norm.weight", {H});
            // (the full-precision stacks are optional: a directory made self-contained keeps only the 8-bit pack)
            if (b.peek(f + "experts.gate_up_proj") || b.peek(f + "experts.down_proj")) {
                b.fl(l.exp_gate_up, f + "experts.gate_up_proj", {E, 2 * MI, LAT});
                b.fl(l.exp_down, f + "experts.down_proj", {E, LAT, MI});
            } else has_src_experts_ = false;
            const std::string e0 = "layers." + std::to_string(L) + ".experts.";
            if (has_pack_ && pb.peek(e0 + "0.w1")) {
                l.exp_w1.resize(size_t(E)); l.exp_w3.resize(size_t(E)); l.exp_w2.resize(size_t(E));
                for (int64_t e = 0; e < E; ++e) {
                    const std::string ep = e0 + std::to_string(e) + ".";
                    pb.u8(l.exp_w1[size_t(e)], ep + "w1", {MI, LAT / 2}, ep + "w1_scale", {MI, LAT / 32});
                    pb.u8(l.exp_w3[size_t(e)], ep + "w3", {MI, LAT / 2}, ep + "w3_scale", {MI, LAT / 32});
                    pb.u8(l.exp_w2[size_t(e)], ep + "w2", {LAT, MI / 2}, ep + "w2_scale", {LAT, MI / 32});
                }
            } else all_packed = false;
        }
        for (size_t gi = 0; gi < c.engram_layer_ids.size(); ++gi) {
            if (c.engram_layer_ids[gi] != L + 1) continue;
            l.engram = int32_t(engrams_.size());
            Hy4fEngram g; g.id = L + 1; g.head_sizes = head_sizes[gi];
            const uint32_t NHd = c.engram_n_heads(); const int64_t HD = c.engram_head_dim(), W = c.engram_width();
            uint64_t rows = 0;
            for (uint64_t s : g.head_sizes) { g.head_offsets.push_back(rows); rows += s; }
            const std::string ge = p + "engram_layer.";
            {   // the table: the exact FP8 file beside the pack when it is there, the source's BF16 tensor otherwise
                const std::string fp = dir + "/engram-" + std::to_string(g.id) + ".fp8";
                struct stat sb{};
                if (::stat(fp.c_str(), &sb) == 0) {
                    if (!c.engram_fp8) { if (b.err.empty()) b.err = fp + " exists but the config does not round the n-gram rows to FP8"; break; }
                    if (uint64_t(sb.st_size) != rows * (4 + uint64_t(HD))) { if (b.err.empty()) b.err = fp + " has " + std::to_string(sb.st_size) + " bytes, " + std::to_string(rows * (4 + uint64_t(HD))) + " expected"; break; }
                    const int fd = ::open(fp.c_str(), O_RDONLY);
                    void* mp = fd < 0 ? MAP_FAILED : ::mmap(nullptr, size_t(sb.st_size), PROT_READ, MAP_SHARED, fd, 0);
                    if (fd >= 0) ::close(fd);
                    if (mp == MAP_FAILED) { if (b.err.empty()) b.err = "cannot map " + fp; break; }
                    ::madvise(mp, size_t(sb.st_size), MADV_RANDOM);
                    maps_.emplace_back(mp, size_t(sb.st_size)); g.fp8 = static_cast<const uint8_t*>(mp); g.fp8_rows = rows;
                } else b.fl(g.table, ge + "multi_head_embedding.embedding.weight", {int64_t(rows), HD});
            }
            const auto* lt = b.peek(ge + "lookup_table");
            if (!lt || lt->shape.size() != 1) { if (b.err.empty()) b.err = "missing or malformed '" + ge + "lookup_table'"; break; }
            g.table_size = uint32_t(lt->shape[0]);
            b.i32(g.lookup, ge + "lookup_table", {int64_t(g.table_size)});
            if (c.engram_pad_id >= g.table_size) { if (b.err.empty()) b.err = "engram_pad_id is outside the lookup table"; break; }
            g.head_norms.resize(NHd);
            for (uint32_t h = 0; h < NHd; ++h) b.fl(g.head_norms[h], ge + "lookup_per_head_norms." + std::to_string(h) + ".weight", {HD});
            mat(g.value_w, ge + "value_proj.weight", H, W);
            b.fl(g.value_b, ge + "value_proj.bias", {H});
            g.key_w.resize(size_t(HC)); g.key_b.resize(size_t(HC)); g.norm_key.resize(size_t(HC)); g.norm_query.resize(size_t(HC)); g.hc_scales.resize(size_t(HC));
            for (int64_t i = 0; i < HC; ++i) {
                const std::string is = std::to_string(i);
                mat(g.key_w[size_t(i)], ge + "key_projs." + is + ".weight", H, W);
                b.fl(g.key_b[size_t(i)], ge + "key_projs." + is + ".bias", {H});
                b.fl(g.norm_key[size_t(i)], ge + "norm_key." + is + ".weight", {H});
                b.fl(g.norm_query[size_t(i)], ge + "norm_query." + is + ".weight", {H});
                b.fl(g.hc_scales[size_t(i)], ge + "unified_per_hc_scales." + is, {H});
            }
            b.fl(g.pre_gamma, ge + "unified_pre_gamma", {H});
            // the hash multipliers: numpy's generator, written by pack.py
            const std::string ids = std::to_string(g.id);
            if (!side.contains("engram_hash_multipliers") || !side["engram_hash_multipliers"].contains(ids)) { if (b.err.empty()) b.err = "hy4flash_ie.json has no hash multipliers for n-gram layer " + ids + " (re-run pack.py)"; break; }
            const json& hm = side["engram_hash_multipliers"][ids];
            if (hm.value("table_size", 0u) != g.table_size || !hm.contains("multipliers") || hm["multipliers"].size() != c.engram_max_ngram) { if (b.err.empty()) b.err = "hy4flash_ie.json: the multipliers of n-gram layer " + ids + " do not fit this checkpoint"; break; }
            for (const auto& mv : hm["multipliers"]) g.multipliers.push_back(mv.get<int64_t>());
            engrams_.push_back(std::move(g));
        }
    }
    if (!b.err.empty()) return b.err;
    if (!pb.err.empty()) return "expert pack: " + pb.err;
    n_overlay_ = b.overlaid;
    if (has_overlay) {                    // every overlay tensor must have replaced something: a name nothing binds is a typo
        if (n_overlay_ != ovl_.tensors().size()) {
            for (const auto& t : ovl_.tensors()) if (!claimed.count(t.name)) return "overlay tensor '" + t.name + "' is not a tensor this model binds";
            return "overlay: " + std::to_string(ovl_.tensors().size()) + " tensors in the file, " + std::to_string(n_overlay_) + " bound";
        }
    }
    has_pack_ = all_packed;

    unclaimed_.clear();
    for (const auto* t : src_.all()) {
        if (claimed.count(t->name)) continue;
        if (t->name.rfind("vit.", 0) == 0 || t->name.rfind("model.dfly_layers.", 0) == 0) continue;   // vision, the drafter: later phases
        unclaimed_.push_back(t->name);
    }
    std::sort(unclaimed_.begin(), unclaimed_.end());
    return {};
}

}  // namespace ie
