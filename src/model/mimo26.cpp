// src/model/mimo26.cpp — MiMo-V2.6 config parse + weight binding (P1, docs/mimo26/00_PORT_PLAN.md).
// Host-only: no device memory, no compute. See include/ie/mimo26.hpp.
#include "ie/mimo26.hpp"

#include "../../third_party/nlohmann/json.hpp"

#include <algorithm>
#include <fstream>
#include <unordered_set>

namespace ie {

using json = nlohmann::json;

namespace {

// Every getter errors rather than defaulting: a config key this engine needs but the file does
// not carry is a bug to surface, not a value to invent. (The V4.1 binder's discipline.)
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
    // A key the checkpoint ships as JSON null: `if_null` stands in ONLY for an explicit null.
    template <class T> void num_or_null(const char* k, T& dst, T if_null) {
        if (const json* v = at(k)) {
            if (v->is_null()) { dst = if_null; return; }
            if (!v->is_number()) { if (err.empty()) err = std::string("config: '") + k + "' is not a number or null"; return; }
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

std::string shape_str(const std::vector<int64_t>& s) {
    std::string o = "[";
    for (size_t i = 0; i < s.size(); ++i) o += (i ? ", " : "") + std::to_string(s[i]);
    return o + "]";
}

int64_t cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// Binds by name, validates shape and dtype string, records the claim. Any failure latches into
// `err` and every later call short-circuits, so the first real problem is the one reported.
class Binder {
public:
    Binder(const SafetensorsModel& m, std::unordered_set<std::string>& claimed)
        : m_(m), claimed_(claimed) {}

    // A plain tensor (BF16 / F32): no scale plane.
    void bind(Ds41Tensor& dst, const std::string& name, std::vector<int64_t> shape, const char* dtype) {
        if (!err.empty()) return;
        dst.w = one(name, shape, dtype);
    }

    // FP8 E4M3 [N, K] with an F32 `<name>_scale_inv` grid of block_n x block_k blocks.
    void bind_fp8(Ds41Tensor& dst, const std::string& name, std::vector<int64_t> shape,
                  uint32_t block_n, uint32_t block_k) {
        if (!err.empty()) return;
        dst.w = one(name, shape, "F8_E4M3");
        if (!err.empty()) return;
        dst.s = one(name + "_scale_inv", {cdiv(shape[0], block_n), cdiv(shape[1], block_k)}, "F32");
    }

    // The fused qkv projection: FP8 [N, K] whose rows were saved in `tp` tensor-parallel shards, so
    // the scale plane has tp * ceil(N / tp / block) block-rows (the llama.cpp converter's detection,
    // candidates 8, 4, then 1 -- the counts differ for every N this model has). Returns tp.
    uint32_t bind_fp8_sharded(Ds41Tensor& dst, const std::string& name, int64_t N, int64_t K, uint32_t block) {
        if (!err.empty()) return 0;
        dst.w = one(name, {N, K}, "F8_E4M3");
        if (!err.empty()) return 0;
        const std::string sn = name + "_scale_inv";
        const auto* s = m_.find(sn);
        if (!s) { err = "missing scale plane '" + sn + "'"; return 0; }
        if (s->dtype_str != "F32" || s->shape.size() != 2 || s->shape[1] != cdiv(K, block)) {
            err = "scale '" + sn + "' is " + s->dtype_str + " " + shape_str(s->shape) + ", expected F32 [rows, " +
                  std::to_string(cdiv(K, block)) + "]";
            return 0;
        }
        for (uint32_t tp : {8u, 4u, 1u}) {
            if (N % tp) continue;
            if (s->shape[0] == int64_t(tp) * cdiv(N / tp, block)) { claimed_.insert(sn); dst.s = s; return tp; }
        }
        err = "scale '" + sn + "' has " + std::to_string(s->shape[0]) + " block-rows, which is no TP in {8, 4, 1} of " +
              std::to_string(N) + " rows at block " + std::to_string(block);
        return 0;
    }

    // An MXFP4 expert plane: nibbles [N, K/2] (the file says U8, read as kI8) with an E8M0 `<name>_scale` [N, K/32].
    void bind_mxfp4(Ds41Tensor& dst, const std::string& name, int64_t N, int64_t K) {
        if (!err.empty()) return;
        dst.w = one(name, {N, K / 2}, "U8");
        if (!err.empty()) return;
        dst.s = one(name + "_scale", {N, K / 32}, "U8");
    }

    // A role the kind says must NOT be present. Binding nothing is the point; a tensor that
    // exists anyway means the kind is wrong, which is worth failing on.
    void forbid(const std::string& name) {
        if (err.empty() && m_.find(name)) err = "tensor '" + name + "' exists but this layer's kind forbids it";
    }

    std::string err;

private:
    const SafeTensorInfo* one(const std::string& name, const std::vector<int64_t>& shape, const char* dtype) {
        const auto* t = m_.find(name);
        if (!t) { err = "missing tensor '" + name + "'"; return nullptr; }
        if (t->shape != shape) {
            err = "tensor '" + name + "' shape " + shape_str(t->shape) + ", expected " + shape_str(shape);
            return nullptr;
        }
        if (dtype && *dtype && t->dtype_str != dtype) {
            err = "tensor '" + name + "' dtype " + t->dtype_str + ", expected " + dtype;
            return nullptr;
        }
        claimed_.insert(name);
        return t;
    }

    const SafetensorsModel& m_;
    std::unordered_set<std::string>& claimed_;
};

bool starts(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

}  // namespace

std::string mimo26_parse_config(const std::string& json_text, Mimo26Config& o) {
    json root;
    try { root = json::parse(json_text); }
    catch (const std::exception& e) { return std::string("config.json parse error: ") + e.what(); }

    if (!root.contains("model_type") || root["model_type"].get<std::string>() != "mimo_v2")
        return "config.json is not model_type mimo_v2";

    Cfg t{root, {}};
    t.num("vocab_size", o.vocab_size);
    t.num("hidden_size", o.dim);
    t.num("intermediate_size", o.inter_dim);
    t.num("moe_intermediate_size", o.moe_inter_dim);
    t.num("num_hidden_layers", o.n_layers);
    if (root.contains("num_nextn_predict_layers")) t.num("num_nextn_predict_layers", o.n_mtp_layers);   // Pro omits it: no MTP
    else o.n_mtp_layers = 0;                                                                           // layers bound (reported unclaimed)
    t.num("num_attention_heads", o.n_heads);
    t.num("num_key_value_heads", o.n_kv_heads);
    t.num("swa_num_key_value_heads", o.n_kv_heads_swa);
    t.num("head_dim", o.head_dim);
    t.num("v_head_dim", o.v_head_dim);
    t.num("sliding_window", o.window);
    t.num("attention_value_scale", o.value_scale);
    t.num("layernorm_epsilon", o.norm_eps);
    t.num("rope_theta", o.rope_theta);
    t.num("swa_rope_theta", o.rope_theta_swa);
    t.num("partial_rotary_factor", o.partial_rotary);
    t.arr("hybrid_layer_pattern", o.hybrid);
    t.arr("moe_layer_freq", o.moe_layer);
    t.num("n_routed_experts", o.n_routed_experts);
    t.num("num_experts_per_tok", o.n_activated_experts);
    t.num("n_group", o.n_group);
    t.num("topk_group", o.topk_group);
    t.str("topk_method", o.topk_method);
    t.str("scoring_func", o.scoring_func);
    t.bl ("norm_topk_prob", o.norm_topk_prob);
    t.num_or_null("routed_scaling_factor", o.route_scale, 1.f);
    t.bl ("add_full_attention_sink_bias", o.sink_full);
    t.bl ("add_swa_attention_sink_bias", o.sink_swa);
    t.num("eos_token_id", o.eos_id);
    t.num("pad_token_id", o.pad_id);
    if (root.contains("image_token_id") && root["image_token_id"].is_number()) o.image_token_id = root["image_token_id"].get<uint32_t>();
    // The SWA layers share the attention geometry (the converter asserts the same three).
    uint32_t swa_hd = 0, swa_heads = 0, swa_vhd = 0;
    t.num("swa_head_dim", swa_hd);
    t.num("swa_num_attention_heads", swa_heads);
    t.num("swa_v_head_dim", swa_vhd);
    if (!t.err.empty()) return t.err;
    if (swa_hd != o.head_dim || swa_heads != o.n_heads || swa_vhd != o.v_head_dim)
        return "config: the SWA attention geometry differs from the full-attention one (unsupported)";

    const auto qc = root.find("quantization_config");
    if (qc == root.end() || !qc->is_object()) return "config: missing quantization_config";
    {
        Cfg q{*qc, {}};
        std::string method, store;
        q.str("quant_method", method);
        q.str("store_dtype", store);
        std::vector<uint32_t> wb;
        q.arr("weight_block_size", wb);
        q.num("mxfp4_block_size", o.mxfp4_block);
        if (!q.err.empty()) return q.err;
        if (method != "fp8" || store != "mxfp4") return "config: quantization_config is not fp8 dense + mxfp4 experts";
        if (wb.size() != 2) return "config: weight_block_size is not [n, k]";
        o.fp8_block_n = wb[0]; o.fp8_block_k = wb[1];
    }

    if (o.hybrid.size() != o.n_layers)    return "config: hybrid_layer_pattern has " + std::to_string(o.hybrid.size()) + " entries, need " + std::to_string(o.n_layers);
    if (o.moe_layer.size() != o.n_layers) return "config: moe_layer_freq has " + std::to_string(o.moe_layer.size()) + " entries, need " + std::to_string(o.n_layers);
    if (o.topk_method != "noaux_tc" || o.scoring_func != "sigmoid") return "config: router is not noaux_tc/sigmoid (unsupported)";
    if (o.n_group != 1 || o.topk_group != 1) return "config: grouped routing (n_group > 1) is unsupported";
    if (o.mxfp4_block != 32) return "config: mxfp4_block_size is not 32";
    if (o.dim % o.fp8_block_k || o.inter_dim % o.fp8_block_k || o.dim % 32 || o.moe_inter_dim % 32)
        return "config: hidden/intermediate sizes are not multiples of the quant blocks";
    if (o.rope_dim() == 0 || o.rope_dim() > o.head_dim || o.rope_dim() % 2) return "config: implausible rope dimension";
    return {};
}

std::string Mimo26Model::load(const std::string& dir) {
    {
        std::ifstream f(dir + "/config.json", std::ios::binary);
        if (!f) return "cannot open " + dir + "/config.json";
        const std::string body((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (auto e = mimo26_parse_config(body, cfg_); !e.empty()) return e;
    }
    if (auto e = store_.open(dir); !e.empty()) return e;

    const auto& c = cfg_;
    const int64_t H = c.dim, FI = c.inter_dim, MI = c.moe_inter_dim, E = c.n_routed_experts;
    const uint32_t n_total = c.n_layers + c.n_mtp_layers;

    std::unordered_set<std::string> claimed;
    claimed.reserve(size_t(store_.all().size()) * 2);
    Binder b(store_, claimed);

    b.bind(embed,      "model.embed_tokens.weight", {int64_t(c.vocab_size), H}, "BF16");
    b.bind(lm_head,    "lm_head.weight",            {int64_t(c.vocab_size), H}, "BF16");
    b.bind(final_norm, "model.norm.weight",         {H}, "BF16");
    if (!b.err.empty()) return b.err;

    layers_.assign(n_total, {});
    for (uint32_t L = 0; L < n_total; ++L) {
        auto& w = layers_[L];
        w.mtp  = c.is_mtp(L);
        w.swa  = c.is_swa(L);
        w.moe  = c.is_moe(L);
        w.has_sink = c.has_sink(L);
        w.n_kv = c.n_kv(L);
        const std::string p = w.mtp ? "model.mtp.layers." + std::to_string(L - c.n_layers) + "."
                                    : "model.layers." + std::to_string(L) + ".";

        b.bind(w.attn_norm, p + "input_layernorm.weight", {H}, "BF16");
        w.qkv_tp = b.bind_fp8_sharded(w.qkv, p + "self_attn.qkv_proj.weight", int64_t(c.qkv_rows(L)), H, c.fp8_block_n);
        b.bind(w.o_proj, p + "self_attn.o_proj.weight", {H, int64_t(c.n_heads) * c.v_head_dim}, "BF16");
        if (w.has_sink) b.bind(w.sink, p + "self_attn.attention_sink_bias", {int64_t(c.n_heads)}, "BF16");
        else            b.forbid(p + "self_attn.attention_sink_bias");

        b.bind(w.ffn_norm, p + (w.mtp ? "pre_mlp_layernorm.weight" : "post_attention_layernorm.weight"), {H}, "BF16");
        if (w.moe) {
            b.bind(w.gate_w,    p + "mlp.gate.weight", {E, H}, "BF16");
            b.bind(w.gate_bias, p + "mlp.gate.e_score_correction_bias", {E}, "F32");
            w.exp_w1.resize(size_t(E)); w.exp_w3.resize(size_t(E)); w.exp_w2.resize(size_t(E));
            for (int64_t x = 0; x < E && b.err.empty(); ++x) {
                const std::string ep = p + "mlp.experts." + std::to_string(x) + ".";
                b.bind_mxfp4(w.exp_w1[size_t(x)], ep + "gate_proj.weight", MI, H);
                b.bind_mxfp4(w.exp_w3[size_t(x)], ep + "up_proj.weight",   MI, H);
                b.bind_mxfp4(w.exp_w2[size_t(x)], ep + "down_proj.weight", H,  MI);
            }
            b.forbid(p + "mlp.gate_proj.weight");
        } else {
            b.bind_fp8(w.mlp_gate, p + "mlp.gate_proj.weight", {FI, H}, c.fp8_block_n, c.fp8_block_k);
            b.bind_fp8(w.mlp_up,   p + "mlp.up_proj.weight",   {FI, H}, c.fp8_block_n, c.fp8_block_k);
            b.bind_fp8(w.mlp_down, p + "mlp.down_proj.weight", {H, FI}, c.fp8_block_n, c.fp8_block_k);
            b.forbid(p + "mlp.gate.weight");
        }
        if (w.mtp) {
            b.bind(w.eh_proj,    p + "eh_proj.weight",          {H, 2 * H}, "BF16");
            b.bind(w.enorm,      p + "enorm.weight",            {H}, "BF16");
            b.bind(w.hnorm,      p + "hnorm.weight",            {H}, "BF16");
            b.bind(w.final_norm, p + "final_layernorm.weight",  {H}, "BF16");
        } else {
            b.forbid(p + "eh_proj.weight");
        }
        if (!b.err.empty()) return "layer " + std::to_string(L) + ": " + b.err;
    }

    // The vision tower, the audio encoder and the speech embeddings: claimed by prefix, nothing in
    // this phase consumes them. Anything else unclaimed is reported.
    for (const auto* t : store_.all()) {
        if (claimed.count(t->name)) continue;
        const auto& n = t->name;
        if (starts(n, "visual.") || starts(n, "audio_encoder.") || starts(n, "speech_embeddings.")) continue;
        unclaimed_.push_back(n);
    }
    std::sort(unclaimed_.begin(), unclaimed_.end());

    // Byte accounting, from the same tensor table the binder used.
    for (const auto* t : store_.all()) {
        const auto& n = t->name;
        uint64_t* bucket = &budget_.norms;
        if (starts(n, "visual."))                                                    bucket = &budget_.vision;
        else if (starts(n, "audio_encoder.") || starts(n, "speech_embeddings."))     bucket = &budget_.audio;
        else if (starts(n, "model.mtp."))                                            bucket = &budget_.mtp;
        else if (n.find(".mlp.experts.") != std::string::npos)                       bucket = &budget_.routed_experts;
        else if (n.find(".mlp.gate.") != std::string::npos)                          bucket = &budget_.routers;
        else if (n.find(".mlp.") != std::string::npos)                               bucket = &budget_.dense_ffn;
        else if (n.find(".self_attn.") != std::string::npos)                         bucket = &budget_.attention;
        else if (n == "model.embed_tokens.weight")                                   bucket = &budget_.embed;
        else if (n == "lm_head.weight")                                              bucket = &budget_.lm_head;
        *bucket += t->nbytes;
    }
    return {};
}

}  // namespace ie
