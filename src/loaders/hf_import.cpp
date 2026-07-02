// src/loaders/hf_import.cpp — HF config.json + tensor-name mapping + the
// AWQ→GGUF import driver. See hf_import.hpp.
#include "ie/hf_import.hpp"
#include "ie/safetensors.hpp"
#include "ie/gguf_writer.hpp"
#include "ie/gguf.hpp"
#include "ie/quantize.hpp"
#include "ie/quant_blocks.hpp"

#include "../../third_party/nlohmann/json.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <list>
#include <string>
#include <vector>

namespace ie {

using json = nlohmann::json;

std::string parse_hf_config(const std::string& config_json, HfModelMeta& out) {
    json c;
    try {
        c = json::parse(config_json);
    } catch (const std::exception& e) {
        return std::string("config.json parse error: ") + e.what();
    }

    auto need_u32 = [&](const char* key, uint32_t& dst) -> bool {
        if (!c.contains(key) || !c[key].is_number()) return false;
        dst = c[key].get<uint32_t>();
        return true;
    };

    if (c.contains("model_type")) out.arch = c["model_type"].get<std::string>();
    DenseConfig& cfg = out.cfg;
    if (!need_u32("num_hidden_layers",   cfg.n_layers))   return "config: missing num_hidden_layers";
    if (!need_u32("hidden_size",         cfg.hidden))     return "config: missing hidden_size";
    if (!need_u32("num_attention_heads", cfg.n_q_heads))  return "config: missing num_attention_heads";
    if (!need_u32("num_key_value_heads", cfg.n_kv_heads)) return "config: missing num_key_value_heads";
    if (!need_u32("intermediate_size",   cfg.ffn))        return "config: missing intermediate_size";
    if (!need_u32("vocab_size",          cfg.vocab))      return "config: missing vocab_size";

    if (!need_u32("head_dim", cfg.head_dim))
        cfg.head_dim = cfg.n_q_heads ? cfg.hidden / cfg.n_q_heads : 0;
    // Qwen3 dense is full-rotary; honor partial_rotary_factor if a model sets it.
    cfg.rope_dim = cfg.head_dim;
    if (c.contains("partial_rotary_factor") && c["partial_rotary_factor"].is_number())
        cfg.rope_dim = uint32_t(cfg.head_dim * c["partial_rotary_factor"].get<double>());

    need_u32("max_position_embeddings", cfg.ctx_train);
    if (c.contains("rope_theta")    && c["rope_theta"].is_number())    cfg.rope_theta = float(c["rope_theta"].get<double>());
    if (c.contains("rms_norm_eps")  && c["rms_norm_eps"].is_number())  cfg.rms_eps    = float(c["rms_norm_eps"].get<double>());
    if (c.contains("tie_word_embeddings")) out.tie_word_embeddings = c["tie_word_embeddings"].get<bool>();
    if (c.contains("attention_bias"))      out.attention_bias      = c["attention_bias"].get<bool>();
    if (c.contains("torch_dtype"))         out.torch_dtype         = c["torch_dtype"].get<std::string>();

    if (c.contains("quantization_config")) {
        out.quantized = true;
        // EXL3 carries no AWQ/GPTQ fields (and bits!=4) → don't run the AWQ parser
        // on it; bits are derived per-tensor from the trellis shape at import.
        const json& qc = c["quantization_config"];
        if (qc.contains("quant_method"))
            out.quant.quant_method = qc["quant_method"].get<std::string>();
        if (out.quant.quant_method == "awq" || out.quant.quant_method == "gptq")
            if (auto m = parse_awq_config(config_json, out.quant); !m.empty()) return m;
    }
    return {};
}

std::string hf_to_gguf_tensor_name(const std::string& n) {
    if (n == "model.embed_tokens.weight") return "token_embd.weight";
    if (n == "model.norm.weight")         return "output_norm.weight";
    if (n == "lm_head.weight")            return "output.weight";

    const std::string pfx = "model.layers.";
    if (n.rfind(pfx, 0) != 0) return {};
    const size_t p = pfx.size();
    const size_t dot = n.find('.', p);
    if (dot == std::string::npos) return {};
    const std::string layer = n.substr(p, dot - p);  // the layer index N
    std::string rest = n.substr(dot + 1);            // role[.suffix]

    // Split off a trailing component suffix (.weight/.qweight/.qzeros/...).
    std::string suffix;
    static const char* sufs[] = {".weight", ".bias", ".qweight", ".qzeros",
                                 ".scales", ".g_idx"};
    for (const char* s : sufs) {
        const size_t pos = rest.rfind(s);
        if (pos != std::string::npos && pos + std::strlen(s) == rest.size()) {
            suffix = s;
            rest   = rest.substr(0, pos);
            break;
        }
    }

    std::string role;
    if      (rest == "self_attn.q_proj")           role = "attn_q";
    else if (rest == "self_attn.k_proj")           role = "attn_k";
    else if (rest == "self_attn.v_proj")           role = "attn_v";
    else if (rest == "self_attn.o_proj")           role = "attn_output";
    else if (rest == "self_attn.q_norm")           role = "attn_q_norm";
    else if (rest == "self_attn.k_norm")           role = "attn_k_norm";
    else if (rest == "mlp.gate_proj")              role = "ffn_gate";
    else if (rest == "mlp.up_proj")                role = "ffn_up";
    else if (rest == "mlp.down_proj")              role = "ffn_down";
    else if (rest == "input_layernorm")            role = "attn_norm";
    else if (rest == "post_attention_layernorm")   role = "ffn_norm";
    else return {};

    return "blk." + layer + "." + role + suffix;
}

namespace {

inline float bf16_to_fp32(uint16_t b) {
    const uint32_t u = uint32_t(b) << 16;
    float f; std::memcpy(&f, &u, 4); return f;
}

// Convert a small tensor (BF16/F16/F32) to an fp32 vector (norms, biases).
std::vector<float> to_f32(const SafeTensorInfo* t) {
    const int64_t n = int64_t(t->numel());
    std::vector<float> out(n);
    const auto* u16 = reinterpret_cast<const uint16_t*>(t->data);
    if (t->dtype == DType::kBF16)      for (int64_t i = 0; i < n; ++i) out[i] = bf16_to_fp32(u16[i]);
    else if (t->dtype == DType::kF16)  for (int64_t i = 0; i < n; ++i) out[i] = fp16_to_fp32(u16[i]);
    else                               std::memcpy(out.data(), t->data, size_t(n) * 4);  // F32
    return out;
}

std::string read_file(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "cannot open " + path;
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return {};
}

// Copy the tokenizer KVs the engine reads from a same-family reference GGUF.
// `vocab_rows` = the emitted embedding row count (config.json vocab_size). When
// the model pads its embedding above the ref tokenizer's token count (Qwen2.5:
// 152064 embed rows vs a 151936-token ref), the token list + token_type arrays
// are padded with UNUSED placeholders up to `vocab_rows` so their length equals
// the embedding. llama.cpp derives n_vocab from the token-list length (NOT the
// .vocab_size KV), so without this the padded embedding is rejected by
// check_tensor_dims. Our own loader reads vocab from the (now-padded) token list
// too → it sees the full vocab, matching the true model (logits over all rows).
void copy_tokenizer_kvs(const GgufReader& ref, GgufWriter& w, uint32_t vocab_rows) {
    auto str = [&](const char* k) {
        if (const auto* kv = ref.find_kv(k); kv && kv->type == GgufValueType::kString)
            w.kv_string(k, std::string(kv->as_string()));
    };
    auto u32 = [&](const char* k) {
        if (const auto* kv = ref.find_kv(k)) w.kv_u32(k, uint32_t(kv->as_int()));
    };
    str("tokenizer.ggml.model");
    str("tokenizer.ggml.pre");
    std::vector<std::string> toks;
    if (const auto* kv = ref.find_kv("tokenizer.ggml.tokens");
        kv && kv->type == GgufValueType::kArray) {
        toks.reserve(kv->n_array);
        for (auto sv : kv->as_string_array()) toks.emplace_back(sv);
    }
    std::vector<int32_t> types;
    if (const auto* kv = ref.find_kv("tokenizer.ggml.token_type");
        kv && kv->type == GgufValueType::kArray) {
        const auto a = kv->as_pod_array<int32_t>();
        types.assign(a.begin(), a.end());
    }
    // pad the token vocabulary up to the embedding row count (never truncate).
    if (vocab_rows > toks.size()) {
        constexpr int32_t kTokenTypeUnused = 5;   // LLAMA_TOKEN_TYPE_UNUSED
        for (size_t i = toks.size(); i < vocab_rows; ++i)
            toks.push_back("[PAD" + std::to_string(i) + "]");
        types.resize(vocab_rows, kTokenTypeUnused);
    }
    if (!toks.empty())  w.kv_string_array("tokenizer.ggml.tokens", toks);
    if (const auto* kv = ref.find_kv("tokenizer.ggml.merges");
        kv && kv->type == GgufValueType::kArray) {
        std::vector<std::string> m;
        m.reserve(kv->n_array);
        for (auto sv : kv->as_string_array()) m.emplace_back(sv);
        w.kv_string_array("tokenizer.ggml.merges", m);
    }
    if (!types.empty()) w.kv_i32_array("tokenizer.ggml.token_type", types);
    u32("tokenizer.ggml.bos_token_id");
    u32("tokenizer.ggml.eos_token_id");
    u32("tokenizer.ggml.padding_token_id");
    if (const auto* kv = ref.find_kv("tokenizer.ggml.add_bos_token"))
        w.kv_bool("tokenizer.ggml.add_bos_token", kv->as_bool());
    str("tokenizer.chat_template");
}

// fp32 row-major [rows][in] → Q6_K blocks appended into `dst` (rows of `in`).
void encode_q6k_rows(const std::vector<float>& flat_rowmajor, int64_t rows, int64_t in,
                     std::vector<uint8_t>& dst) {
    const int64_t bpr = in / 256;
    dst.resize(size_t(rows) * bpr * sizeof(block_q6_K));
    auto* blk = reinterpret_cast<block_q6_K*>(dst.data());
    for (int64_t r = 0; r < rows; ++r)
        quantize_row_q6_K(flat_rowmajor.data() + r * in, blk + r * bpr, in);
}

// Same, Q4_K (≈33% smaller; used for projections — the bulk of the weights).
void encode_q4k_rows(const std::vector<float>& flat_rowmajor, int64_t rows, int64_t in,
                     std::vector<uint8_t>& dst) {
    const int64_t bpr = in / 256;
    dst.resize(size_t(rows) * bpr * sizeof(block_q4_K));
    auto* blk = reinterpret_cast<block_q4_K*>(dst.data());
    for (int64_t r = 0; r < rows; ++r)
        quantize_row_q4_K(flat_rowmajor.data() + r * in, blk + r * bpr, in);
}

}  // namespace

std::string import_awq_to_gguf(const std::string& hf_dir, const std::string& out_gguf,
                               const std::string& tokenizer_ref_gguf, std::string* log) {
    auto note = [&](const std::string& s) { if (log) *log += s + "\n"; };

    std::string cfg_json;
    if (auto e = read_file(hf_dir + "/config.json", cfg_json); !e.empty()) return e;
    HfModelMeta meta;
    if (auto e = parse_hf_config(cfg_json, meta); !e.empty()) return e;
    if (!meta.quantized ||
        (meta.quant.quant_method != "awq" && meta.quant.quant_method != "gptq"))
        return "not an AWQ/GPTQ checkpoint (quant_method must be awq or gptq)";
    if (meta.arch != "qwen3" && meta.arch != "qwen2")
        return "import supports model_type 'qwen3'/'qwen2' (got '" + meta.arch + "')";
    const int64_t G = meta.quant.group_size;
    note("config: " + meta.arch + " dense, " + std::to_string(meta.cfg.n_layers) + " layers, hidden " +
         std::to_string(meta.cfg.hidden) + ", " + meta.quant.quant_method + "/" +
         std::to_string(G) + (meta.quant.desc_act ? " desc_act" : ""));

    SafetensorsModel st;
    if (auto e = st.open(hf_dir); !e.empty()) return "safetensors: " + e;
    note("safetensors: " + std::to_string(st.shard_count()) + " shard(s)");
    GgufReader ref;
    if (auto e = ref.open(tokenizer_ref_gguf); !e.empty()) return "tokenizer ref: " + e;

    GgufWriter w;
    const std::string A = meta.arch;
    w.kv_string("general.architecture", A);
    w.kv_string("general.name", "imported-awq");
    w.kv_u32(A + ".block_count", meta.cfg.n_layers);
    w.kv_u32(A + ".context_length", meta.cfg.ctx_train);
    // vocab_size = the embedding row count (config.json vocab_size). Qwen2.5 pads
    // this above the tokenizer token count (e.g. 72B: 152064 embed rows vs 151936
    // tokens). Without this KV, llama.cpp derives n_vocab from the token-list
    // length and then REJECTS our padded embedding tensor (check_tensor_dims:
    // "token_embd.weight has wrong shape"). Declaring it = the emitted embedding
    // rows makes the GGUF round-trip to llama.cpp (stock Qwen2.5 convention). Our
    // own loader reads vocab from the token list (model_config.cpp), so this is a
    // no-op for the engine — purely an interop fix.
    w.kv_u32(A + ".vocab_size", meta.cfg.vocab);
    w.kv_u32(A + ".embedding_length", meta.cfg.hidden);
    w.kv_u32(A + ".feed_forward_length", meta.cfg.ffn);
    w.kv_u32(A + ".attention.head_count", meta.cfg.n_q_heads);
    w.kv_u32(A + ".attention.head_count_kv", meta.cfg.n_kv_heads);
    w.kv_u32(A + ".attention.key_length", meta.cfg.head_dim);
    w.kv_u32(A + ".attention.value_length", meta.cfg.head_dim);
    w.kv_u32(A + ".rope.dimension_count", meta.cfg.rope_dim);
    w.kv_f32(A + ".rope.freq_base", meta.cfg.rope_theta);
    w.kv_f32(A + ".attention.layer_norm_rms_epsilon", meta.cfg.rms_eps);
    copy_tokenizer_kvs(ref, w, meta.cfg.vocab);

    // Default (accumulate) path: tensor buffers must outlive w.write(), so a
    // list keeps stable addresses. Streaming path (IE_IMPORT_STREAM=1) keeps only
    // ONE tensor's bytes in RAM — REQUIRED for the 72B (~40 GB output won't fit
    // 30 GB RAM otherwise). Both paths produce a byte-identical GGUF.
    std::list<std::vector<uint8_t>> hold;
    int n_proj = 0, n_norm = 0, n_embed = 0, n_bias = 0;

    const bool stream = std::getenv("IE_IMPORT_STREAM") != nullptr;

    // Classify a safetensors tensor → its output GGUF spec (gname, dtype, shape,
    // nbytes) WITHOUT quantizing. Must mirror the emit sites in the loop below;
    // a mismatch is caught by stream_next's nbytes assertion (not silent).
    struct OutSpec { bool ok = false; std::string gname; DType dtype;
                     std::vector<uint64_t> shape; uint64_t nbytes = 0; };
    auto classify_out = [&](const auto& t) -> OutSpec {
        OutSpec s;
        const std::string& hf = t.name;
        auto ends = [&](const char* suf, size_t n) {
            return hf.size() > n && hf.compare(hf.size() - n, n, suf) == 0; };
        if (ends(".qzeros", 7) || ends(".scales", 7) || ends(".g_idx", 6)) return s;
        if (ends(".qweight", 8)) {
            s.gname = hf_to_gguf_tensor_name(hf.substr(0, hf.size() - 8) + ".weight");
            if (s.gname.empty()) return s;
            const bool is_gptq = (meta.quant.quant_method == "gptq");
            const int64_t in  = is_gptq ? t.shape[0] * 8 : t.shape[0];
            const int64_t out = is_gptq ? t.shape[1]     : t.shape[1] * 8;
            s.dtype = DType::kQ4_K; s.shape = {uint64_t(in), uint64_t(out)};
            s.nbytes = uint64_t(out) * (uint64_t(in) / 256) * sizeof(block_q4_K);
            s.ok = true; return s;
        }
        const bool is_weight = ends(".weight", 7), is_bias = ends(".bias", 5);
        if (is_bias || (is_weight && t.shape.size() == 1)) {
            s.gname = hf_to_gguf_tensor_name(hf);
            if (s.gname.empty()) return s;
            s.dtype = DType::kF32; s.shape = {uint64_t(t.shape[0])};
            s.nbytes = uint64_t(t.shape[0]) * sizeof(float); s.ok = true; return s;
        }
        if (is_weight) {
            s.gname = hf_to_gguf_tensor_name(hf);
            if (s.gname.empty()) return s;
            const int64_t rows = t.shape[0], in = t.shape[1];
            s.dtype = DType::kQ6_K; s.shape = {uint64_t(in), uint64_t(rows)};
            s.nbytes = uint64_t(rows) * (uint64_t(in) / 256) * sizeof(block_q6_K);
            s.ok = true; return s;
        }
        return s;
    };

    // Emit one finished tensor: stream it (free immediately) or accumulate it.
    auto emit = [&](const std::string& gname, DType dtype,
                    std::vector<uint64_t> shape, std::vector<uint8_t>&& buf) -> std::string {
        if (stream) return w.stream_next(buf.data(), buf.size());
        hold.emplace_back(std::move(buf));
        w.tensor(gname, dtype, shape, hold.back().data(), hold.back().size());
        return {};
    };

    if (stream) {
        for (const auto* tp : st.all()) {
            auto s = classify_out(*tp);
            if (s.ok) w.tensor_info(s.gname, s.dtype, s.shape, s.nbytes);
        }
        if (auto e = w.begin_streaming(out_gguf); !e.empty()) return "begin_streaming: " + e;
        note("streaming import (RAM-safe: one tensor at a time)");
    }

    for (const auto* tp : st.all()) {
        const auto& t = *tp;
        const std::string& hf = t.name;
        if (hf.size() > 7 && hf.compare(hf.size() - 7, 7, ".qzeros") == 0) continue;
        if (hf.size() > 7 && hf.compare(hf.size() - 7, 7, ".scales") == 0) continue;
        if (hf.size() > 6 && hf.compare(hf.size() - 6, 6, ".g_idx") == 0)  continue;

        // --- AWQ/GPTQ-quantized projection (process on the .qweight member) ---
        if (hf.size() > 8 && hf.compare(hf.size() - 8, 8, ".qweight") == 0) {
            const std::string base = hf.substr(0, hf.size() - 8);
            const std::string gname = hf_to_gguf_tensor_name(base + ".weight");
            if (gname.empty()) { note("skip unknown proj: " + hf); continue; }
            const auto* qw = st.find(base + ".qweight");
            const auto* qz = st.find(base + ".qzeros");
            const auto* sc = st.find(base + ".scales");
            if (!qw || !qz || !sc) return "quant triplet incomplete for " + base;

            const bool is_gptq = (meta.quant.quant_method == "gptq");
            // qweight layout: AWQ [in, out/8] (output-axis pack); GPTQ [in/8, out].
            const int64_t in     = is_gptq ? qw->shape[0] * 8 : qw->shape[0];
            const int64_t out    = is_gptq ? qw->shape[1]     : qw->shape[1] * 8;
            const int64_t groups = sc->shape[0];
            if (in % 256 != 0) return "in_features not /256 for " + base;
            const int64_t gs = in / groups;

            // Alignment-safe copies out of the mmap (by exact byte count).
            std::vector<int32_t>  qwb(qw->nbytes / 4), qzb(qz->nbytes / 4);
            std::vector<uint16_t> scb(sc->nbytes / 2), wf(size_t(in) * out);
            std::memcpy(qwb.data(), qw->data, qw->nbytes);
            std::memcpy(qzb.data(), qz->data, qz->nbytes);
            std::memcpy(scb.data(), sc->data, sc->nbytes);

            std::string de;
            if (is_gptq) {
                std::vector<int32_t> gidxb;
                const int32_t* gidx = nullptr;   // desc_act=false → g_idx == i/gs
                if (meta.quant.desc_act)
                    if (const auto* gi = st.find(base + ".g_idx")) {
                        gidxb.resize(gi->nbytes / 4);
                        std::memcpy(gidxb.data(), gi->data, gi->nbytes);
                        gidx = gidxb.data();
                    }
                de = gptq_dequant_to_fp16(qwb.data(), qzb.data(), scb.data(), gidx,
                                          in, out, gs, wf.data());
            } else {
                de = awq_dequant_to_fp16(qwb.data(), qzb.data(), scb.data(),
                                         in, out, gs, wf.data());
            }
            if (!de.empty()) return "dequant " + base + ": " + de;

            std::vector<float> colmajor(size_t(out) * in);   // [out][in] (transpose)
            for (int64_t o = 0; o < out; ++o)
                for (int64_t i = 0; i < in; ++i)
                    colmajor[size_t(o) * in + i] = fp16_to_fp32(wf[size_t(i) * out + o]);
            std::vector<uint8_t> buf;
            encode_q4k_rows(colmajor, out, in, buf);           // Q4_K — projections
            if (auto e = emit(gname, DType::kQ4_K, {uint64_t(in), uint64_t(out)}, std::move(buf)); !e.empty())
                return "emit " + gname + ": " + e;
            ++n_proj;
            continue;
        }

        // --- 1-D F32 params: norms (.weight) and attention biases (.bias, Qwen2) ---
        const bool is_weight = (hf.size() > 7 && hf.compare(hf.size() - 7, 7, ".weight") == 0);
        const bool is_bias   = (hf.size() > 5 && hf.compare(hf.size() - 5, 5, ".bias") == 0);
        if (is_bias || (is_weight && t.shape.size() == 1)) {
            const std::string gname = hf_to_gguf_tensor_name(hf);
            if (gname.empty()) { note("skip unknown 1d: " + hf); continue; }
            const auto f = to_f32(&t);
            std::vector<uint8_t> buf(f.size() * sizeof(float));
            std::memcpy(buf.data(), f.data(), f.size() * sizeof(float));
            if (auto e = emit(gname, DType::kF32, {uint64_t(f.size())}, std::move(buf)); !e.empty())
                return "emit " + gname + ": " + e;
            (is_bias ? n_bias : n_norm)++;
            continue;
        }

        // --- embeddings / lm_head [vocab, hidden] → Q6_K (dtype-aware: BF16/F16/F32) ---
        if (is_weight) {
            const std::string gname = hf_to_gguf_tensor_name(hf);
            if (gname.empty()) { note("skip unknown tensor: " + hf); continue; }
            const int64_t rows = t.shape[0], in = t.shape[1];
            if (in % 256 != 0) return "embedding dim not /256";
            std::vector<float> rm = to_f32(&t);   // handles BF16 (Qwen3) and F16 (Qwen2)
            std::vector<uint8_t> buf;
            encode_q6k_rows(rm, rows, in, buf);
            // GGUF ne = {hidden, vocab}; data is row(token)-major, in contiguous.
            if (auto e = emit(gname, DType::kQ6_K, {uint64_t(in), uint64_t(rows)}, std::move(buf)); !e.empty())
                return "emit " + gname + ": " + e;
            ++n_embed;
            continue;
        }
        note("skip unrecognized: " + hf);
    }

    note("encoded: " + std::to_string(n_proj) + " projections, " +
         std::to_string(n_norm) + " norms, " + std::to_string(n_bias) + " biases, " +
         std::to_string(n_embed) + " embedding(s)");
    if (stream) { if (auto e = w.end_streaming();  !e.empty()) return "end_streaming: " + e; }
    else        { if (auto e = w.write(out_gguf);   !e.empty()) return "write: " + e; }
    note(std::string(stream ? "streamed " : "wrote ") + out_gguf);
    return {};
}

std::string import_exl3_to_gguf(const std::string& hf_dir, const std::string& out_gguf,
                               const std::string& tokenizer_ref_gguf, std::string* log) {
    auto note = [&](const std::string& s) { if (log) *log += s + "\n"; };

    std::string cfg_json;
    if (auto e = read_file(hf_dir + "/config.json", cfg_json); !e.empty()) return e;
    HfModelMeta meta;
    if (auto e = parse_hf_config(cfg_json, meta); !e.empty()) return e;
    if (!meta.quantized || meta.quant.quant_method != "exl3")
        return "not an EXL3 checkpoint (quant_method must be exl3)";
    if (meta.arch != "llama")
        return "exl3 import v1 supports model_type 'llama' (got '" + meta.arch + "')";
    note("config: llama dense, " + std::to_string(meta.cfg.n_layers) + " layers, hidden " +
         std::to_string(meta.cfg.hidden) + ", exl3");

    // llama3 RoPE scaling (rope_freqs.weight is COMPUTED — the 3B ref can't be
    // copied, different head_dim). Plain rope (no scaling) → skip the tensor.
    bool   llama3_rope = false;
    double rope_factor = 8.0, low_ff = 1.0, high_ff = 4.0;
    uint32_t orig_ctx  = 8192;
    {
        json c = json::parse(cfg_json);
        if (c.contains("rope_scaling") && c["rope_scaling"].is_object()) {
            const auto& rs = c["rope_scaling"];
            const std::string rtype =
                rs.contains("rope_type") ? rs["rope_type"].get<std::string>()
              : rs.contains("type")      ? rs["type"].get<std::string>() : std::string();
            if (rtype == "llama3") {
                llama3_rope = true;
                if (rs.contains("factor"))           rope_factor = rs["factor"].get<double>();
                if (rs.contains("low_freq_factor"))  low_ff      = rs["low_freq_factor"].get<double>();
                if (rs.contains("high_freq_factor")) high_ff     = rs["high_freq_factor"].get<double>();
                if (rs.contains("original_max_position_embeddings"))
                    orig_ctx = rs["original_max_position_embeddings"].get<uint32_t>();
            }
        }
    }

    SafetensorsModel st;
    if (auto e = st.open(hf_dir); !e.empty()) return "safetensors: " + e;
    note("safetensors: " + std::to_string(st.shard_count()) + " shard(s)");
    GgufReader ref;
    if (auto e = ref.open(tokenizer_ref_gguf); !e.empty()) return "tokenizer ref: " + e;

    GgufWriter w;
    const std::string A = "llama";
    w.kv_string("general.architecture", A);
    w.kv_string("general.name", "imported-exl3");
    w.kv_u32(A + ".block_count", meta.cfg.n_layers);
    w.kv_u32(A + ".context_length", meta.cfg.ctx_train);
    w.kv_u32(A + ".vocab_size", meta.cfg.vocab);
    w.kv_u32(A + ".embedding_length", meta.cfg.hidden);
    w.kv_u32(A + ".feed_forward_length", meta.cfg.ffn);
    w.kv_u32(A + ".attention.head_count", meta.cfg.n_q_heads);
    w.kv_u32(A + ".attention.head_count_kv", meta.cfg.n_kv_heads);
    w.kv_u32(A + ".attention.key_length", meta.cfg.head_dim);
    w.kv_u32(A + ".attention.value_length", meta.cfg.head_dim);
    w.kv_u32(A + ".rope.dimension_count", meta.cfg.rope_dim);
    w.kv_f32(A + ".rope.freq_base", meta.cfg.rope_theta);
    w.kv_f32(A + ".attention.layer_norm_rms_epsilon", meta.cfg.rms_eps);
    copy_tokenizer_kvs(ref, w, meta.cfg.vocab);

    std::list<std::vector<uint8_t>> hold;
    auto copy_bytes = [&](const void* p, size_t n) {
        std::vector<uint8_t> b(n); std::memcpy(b.data(), p, n); return b; };
    auto emit = [&](const std::string& gname, DType dtype,
                    std::vector<uint64_t> shape, std::vector<uint8_t>&& buf) {
        hold.emplace_back(std::move(buf));
        w.tensor(gname, dtype, shape, hold.back().data(), hold.back().size());
    };

    // rope_freqs.weight (llama3 scaling) — exactly llama.cpp's per-pair factor
    // (ggml: inv_freq /= factor): 1 at high freq, `factor` at low freq, smooth between.
    if (llama3_rope) {
        const uint32_t half = meta.cfg.rope_dim / 2;
        const double base = meta.cfg.rope_theta;
        const double low_wl = double(orig_ctx) / low_ff, high_wl = double(orig_ctx) / high_ff;
        constexpr double PI = 3.14159265358979323846;
        std::vector<float> rf(half);
        for (uint32_t r = 0; r < half; ++r) {
            const double inv = std::pow(base, -double(2 * r) / double(meta.cfg.rope_dim));
            const double wl  = 2.0 * PI / inv;
            double f;
            if      (wl < high_wl) f = 1.0;
            else if (wl > low_wl)  f = rope_factor;
            else { const double s = (double(orig_ctx) / wl - low_ff) / (high_ff - low_ff);
                   f = 1.0 / ((1.0 - s) / rope_factor + s); }
            rf[r] = float(f);
        }
        std::vector<uint8_t> buf(rf.size() * sizeof(float));
        std::memcpy(buf.data(), rf.data(), buf.size());
        emit("rope_freqs.weight", DType::kF32, {uint64_t(half)}, std::move(buf));
        note("rope_freqs: llama3, " + std::to_string(half) + " factors (factor " +
             std::to_string(rope_factor) + ")");
    }

    int n_exl3 = 0, n_norm = 0, n_embed = 0;
    for (const auto* tp : st.all()) {
        const auto& t = *tp;
        const std::string& hf = t.name;
        auto ends = [&](const char* s, size_t n) {
            return hf.size() > n && hf.compare(hf.size() - n, n, s) == 0; };

        if (ends(".suh", 4) || ends(".svh", 4)) continue;   // emitted with .trellis

        if (ends(".trellis", 8)) {
            const std::string base = hf.substr(0, hf.size() - 8);
            const std::string gw = hf_to_gguf_tensor_name(base + ".weight");
            if (gw.empty()) { note("skip unknown proj: " + hf); continue; }
            const std::string groot = gw.substr(0, gw.size() - 7);   // strip ".weight"
            if (t.shape.size() != 3) return "trellis not 3D: " + hf;
            const int64_t TK = t.shape[0], TN = t.shape[1], wbits = t.shape[2];
            if (wbits % 16 != 0) return "trellis last dim not /16: " + hf;
            // verbatim EXL3 codes; GGUF ne = {16*bits, N/16, K/16}.
            emit(gw, DType::kEXL3, {uint64_t(wbits), uint64_t(TN), uint64_t(TK)},
                 copy_bytes(t.data, t.nbytes));
            const auto* su = st.find(base + ".suh");
            const auto* sv = st.find(base + ".svh");
            if (!su || !sv) return "exl3 missing suh/svh for " + base;
            emit(groot + ".suh", DType::kF16, {uint64_t(su->shape[0])}, copy_bytes(su->data, su->nbytes));
            emit(groot + ".svh", DType::kF16, {uint64_t(sv->shape[0])}, copy_bytes(sv->data, sv->nbytes));
            ++n_exl3;
            continue;
        }

        if (hf == "model.embed_tokens.weight") {
            if (t.dtype != DType::kF16)
                return "exl3 import: token_embd expected F16 (got " +
                       std::string(type_name(t.dtype)) + ")";
            // verbatim F16; GGUF ne = {hidden, vocab}, token-major rows.
            emit("token_embd.weight", DType::kF16,
                 {uint64_t(t.shape[1]), uint64_t(t.shape[0])}, copy_bytes(t.data, t.nbytes));
            ++n_embed;
            continue;
        }

        if (ends(".weight", 7) && t.shape.size() == 1) {     // norms → F32
            const std::string gname = hf_to_gguf_tensor_name(hf);
            if (gname.empty()) { note("skip unknown 1d: " + hf); continue; }
            const auto f = to_f32(&t);
            std::vector<uint8_t> buf(f.size() * sizeof(float));
            std::memcpy(buf.data(), f.data(), buf.size());
            emit(gname, DType::kF32, {uint64_t(f.size())}, std::move(buf));
            ++n_norm;
            continue;
        }
        note("skip unrecognized: " + hf);
    }

    note("encoded: " + std::to_string(n_exl3) + " EXL3 linears, " +
         std::to_string(n_norm) + " norms, " + std::to_string(n_embed) + " embedding(s)");
    if (auto e = w.write(out_gguf); !e.empty()) return "write: " + e;
    note("wrote " + out_gguf);
    return {};
}

// ===================================================================== //
// EXL3 Qwen3-Next (MoE + DeltaNet) importer — fused expert banks, streamed.
// Spec: MASTER_DEV_PLAN.md §7 "Phase D — RECON RESOLVED". The reference GGUF
// contract is the bartowski/Huihui Q4_K_M qwen3next (diff target).
// ===================================================================== //
std::string import_exl3_qwen3next_to_gguf(const std::string& hf_dir, const std::string& out_gguf,
                                          const std::string& tokenizer_ref_gguf, std::string* log) {
    auto note = [&](const std::string& s) { if (log) *log += s + "\n"; };

    std::string cfg_json;
    if (auto e = read_file(hf_dir + "/config.json", cfg_json); !e.empty()) return e;
    HfModelMeta meta;
    if (auto e = parse_hf_config(cfg_json, meta); !e.empty()) return e;
    if (!meta.quantized || meta.quant.quant_method != "exl3")
        return "not an EXL3 checkpoint (quant_method must be exl3)";
    if (meta.arch != "qwen3_next" && meta.arch != "qwen3next")
        return "exl3 qwen3next import expects model_type 'qwen3_next' (got '" + meta.arch + "')";

    json c = json::parse(cfg_json);
    auto u = [&](const char* k, uint32_t dflt) -> uint32_t {
        return c.contains(k) && c[k].is_number() ? c[k].get<uint32_t>() : dflt; };
    const uint32_t n_layers     = u("num_hidden_layers", 48);
    const uint32_t hidden       = u("hidden_size", 2048);
    const uint32_t n_q          = u("num_attention_heads", 16);
    const uint32_t n_kv         = u("num_key_value_heads", 2);
    const uint32_t head_dim     = u("head_dim", 256);
    const uint32_t n_expert     = u("num_experts", 512);
    const uint32_t n_exp_used   = u("num_experts_per_tok", 10);
    const uint32_t moe_inter    = u("moe_intermediate_size", 512);
    const uint32_t shared_inter = u("shared_expert_intermediate_size", 512);
    const uint32_t ffn_len      = u("intermediate_size", 5120);
    const uint32_t fa_interval  = u("full_attention_interval", 4);
    const uint32_t vocab        = u("vocab_size", 151936);
    const uint32_t ctx_len      = u("max_position_embeddings", 262144);
    const float    rope_theta   = c.contains("rope_theta")   ? c["rope_theta"].get<float>()   : 1e7f;
    const float    rms_eps      = c.contains("rms_norm_eps") ? c["rms_norm_eps"].get<float>() : 1e-6f;
    const double   partial      = c.contains("partial_rotary_factor")
                                  ? c["partial_rotary_factor"].get<double>() : 0.25;
    const uint32_t rope_dim     = uint32_t(head_dim * partial);   // 64
    const uint32_t lin_v_heads  = u("linear_num_value_heads", 32);
    const uint32_t lin_v_dim    = u("linear_value_head_dim", 128);
    const uint32_t lin_k_heads  = u("linear_num_key_heads", 16);
    const uint32_t lin_k_dim    = u("linear_key_head_dim", 128);
    const uint32_t conv_kernel  = u("linear_conv_kernel_dim", 4);
    const uint32_t ssm_inner    = lin_v_heads * lin_v_dim;        // 4096
    const uint32_t ssm_state    = lin_k_dim;                      // 128
    const uint32_t ssm_groups   = lin_k_heads;                    // 16
    const uint32_t ssm_dt_rank  = lin_v_heads;                    // 32
    note("config: qwen3_next, " + std::to_string(n_layers) + " layers, hidden " +
         std::to_string(hidden) + ", " + std::to_string(n_expert) + " experts, exl3");

    SafetensorsModel st;
    if (auto e = st.open(hf_dir); !e.empty()) return "safetensors: " + e;
    note("safetensors: " + std::to_string(st.shard_count()) + " shard(s)");
    GgufReader ref;
    if (auto e = ref.open(tokenizer_ref_gguf); !e.empty()) return "tokenizer ref: " + e;

    GgufWriter w;
    const std::string A = "qwen3next";
    w.kv_string("general.architecture", A);
    w.kv_string("general.name", "imported-exl3-qwen3next");
    w.kv_u32(A + ".block_count", n_layers);
    w.kv_u32(A + ".context_length", ctx_len);
    w.kv_u32(A + ".embedding_length", hidden);
    w.kv_u32(A + ".feed_forward_length", ffn_len);
    w.kv_u32(A + ".attention.head_count", n_q);
    w.kv_u32(A + ".attention.head_count_kv", n_kv);
    w.kv_u32(A + ".attention.key_length", head_dim);
    w.kv_u32(A + ".attention.value_length", head_dim);
    w.kv_f32(A + ".attention.layer_norm_rms_epsilon", rms_eps);
    w.kv_u32(A + ".expert_count", n_expert);
    w.kv_u32(A + ".expert_used_count", n_exp_used);
    w.kv_u32(A + ".expert_feed_forward_length", moe_inter);
    w.kv_u32(A + ".expert_shared_feed_forward_length", shared_inter);
    w.kv_f32(A + ".rope.freq_base", rope_theta);
    w.kv_u32(A + ".rope.dimension_count", rope_dim);
    w.kv_u32(A + ".full_attention_interval", fa_interval);
    w.kv_u32(A + ".ssm.state_size", ssm_state);
    w.kv_u32(A + ".ssm.conv_kernel", conv_kernel);
    w.kv_u32(A + ".ssm.group_count", ssm_groups);
    w.kv_u32(A + ".ssm.time_step_rank", ssm_dt_rank);
    w.kv_u32(A + ".ssm.inner_size", ssm_inner);
    copy_tokenizer_kvs(ref, w, vocab);

    // ----- ordered output-tensor list (two-pass streaming, RAM-safe) -----
    // CastF32P1 = norm.weight + 1 (qwen3next RMSNorm convention, all norms except linear_attn.norm);
    // CastNegExp = ssm_a = -exp(A_log). Both match llama.cpp's Qwen3NextModel.modify_tensors.
    enum class Kind { Verbatim, CastF32, CastF32P1, CastNegExp, Fused };
    struct Out {
        std::string name; DType dt; std::vector<uint64_t> shape; uint64_t nbytes;
        Kind kind; const SafeTensorInfo* src = nullptr;
        std::vector<const SafeTensorInfo*> srcs;
    };
    std::vector<Out> outs;
    std::string err;

    auto need = [&](const std::string& nm) -> const SafeTensorInfo* {
        const auto* t = st.find(nm);
        if (!t && err.empty()) err = "missing source tensor: " + nm;
        return t;
    };
    auto add_verbatim = [&](const std::string& gname, DType dt,
                            std::vector<uint64_t> shape, const SafeTensorInfo* t) {
        if (!t) return;
        outs.push_back({gname, dt, std::move(shape), t->nbytes, Kind::Verbatim, t, {}});
    };
    auto add_cast_f32 = [&](const std::string& gname, std::vector<uint64_t> shape,
                            const SafeTensorInfo* t) {
        if (!t) return;
        outs.push_back({gname, DType::kF32, std::move(shape), uint64_t(t->numel()) * 4,
                        Kind::CastF32, t, {}});
    };
    // RMSNorm weight + 1 (every qwen3next norm.weight EXCEPT linear_attn.norm/ssm_norm).
    auto add_norm = [&](const std::string& gname, std::vector<uint64_t> shape,
                        const SafeTensorInfo* t) {
        if (!t) return;
        outs.push_back({gname, DType::kF32, std::move(shape), uint64_t(t->numel()) * 4,
                        Kind::CastF32P1, t, {}});
    };
    // ssm_a = -exp(A_log) (the DeltaNet state-decay; folded at convert time, not forward).
    auto add_negexp = [&](const std::string& gname, std::vector<uint64_t> shape,
                          const SafeTensorInfo* t) {
        if (!t) return;
        outs.push_back({gname, DType::kF32, std::move(shape), uint64_t(t->numel()) * 4,
                        Kind::CastNegExp, t, {}});
    };
    // One EXL3 linear: <hf_base>.{trellis,suh,svh} → <groot>.weight/.suh/.svh.
    // Source trellis numpy shape = [K/16, N/16, 16*bits]; ggml ne = [16*bits, N/16, K/16]
    // (reverse dim order, bytes verbatim). bits = trellis.shape[2]/16.
    auto add_exl3 = [&](const std::string& groot, const std::string& hf_base) {
        const auto* tr = need(hf_base + ".trellis");
        const auto* su = need(hf_base + ".suh");
        const auto* sv = need(hf_base + ".svh");
        if (!tr || !su || !sv) return;
        if (tr->shape.size() != 3) { if (err.empty()) err = "trellis not 3D: " + hf_base; return; }
        const uint64_t TK = tr->shape[0], TN = tr->shape[1], wbits = tr->shape[2];
        add_verbatim(groot + ".weight", DType::kEXL3, {wbits, TN, TK}, tr);
        add_verbatim(groot + ".suh", DType::kF16, {uint64_t(su->shape[0])}, su);
        add_verbatim(groot + ".svh", DType::kF16, {uint64_t(sv->shape[0])}, sv);
    };
    // Fused expert bank: all E experts' {trellis,suh,svh} → 3 banks. Experts are the
    // OUTERMOST ggml dim (contiguous per-expert slab; loader stride = nbytes/E).
    auto add_experts = [&](uint32_t L, const std::string& hf_proj, const std::string& g_exps) {
        const std::string base = "model.layers." + std::to_string(L) + ".mlp.experts.";
        std::vector<const SafeTensorInfo*> trs(n_expert), sus(n_expert), svs(n_expert);
        for (uint32_t e = 0; e < n_expert; ++e) {
            trs[e] = need(base + std::to_string(e) + "." + hf_proj + ".trellis");
            sus[e] = need(base + std::to_string(e) + "." + hf_proj + ".suh");
            svs[e] = need(base + std::to_string(e) + "." + hf_proj + ".svh");
        }
        if (!trs[0] || !sus[0] || !svs[0]) return;
        if (trs[0]->shape.size() != 3) { if (err.empty()) err = "expert trellis not 3D"; return; }
        const uint64_t TK = trs[0]->shape[0], TN = trs[0]->shape[1], wbits = trs[0]->shape[2];
        for (uint32_t e = 1; e < n_expert; ++e)
            if (trs[e] && trs[e]->shape[2] != int64_t(wbits) && err.empty()) {
                err = "non-uniform expert bits: " + g_exps + " L" + std::to_string(L); return; }
        const std::string root = "blk." + std::to_string(L) + "." + g_exps;
        outs.push_back({root + ".weight", DType::kEXL3, {wbits, TN, TK, n_expert},
                        trs[0]->nbytes * n_expert, Kind::Fused, nullptr, std::move(trs)});
        outs.push_back({root + ".suh", DType::kF16, {uint64_t(sus[0]->shape[0]), n_expert},
                        sus[0]->nbytes * n_expert, Kind::Fused, nullptr, std::move(sus)});
        outs.push_back({root + ".svh", DType::kF16, {uint64_t(svs[0]->shape[0]), n_expert},
                        svs[0]->nbytes * n_expert, Kind::Fused, nullptr, std::move(svs)});
    };

    // ---- global tensors ----
    add_verbatim("token_embd.weight", DType::kF16, {hidden, vocab}, need("model.embed_tokens.weight"));
    add_norm("output_norm.weight", {hidden}, need("model.norm.weight"));
    add_exl3("output", "lm_head");

    // ---- layers ----
    for (uint32_t L = 0; L < n_layers; ++L) {
        const std::string p  = "blk." + std::to_string(L) + ".";
        const std::string hl = "model.layers." + std::to_string(L) + ".";
        const bool full = ((L + 1) % fa_interval) == 0;

        add_norm(p + "attn_norm.weight", {hidden}, need(hl + "input_layernorm.weight"));
        add_norm(p + "post_attention_norm.weight", {hidden},
                 need(hl + "post_attention_layernorm.weight"));

        if (!full) {  // DeltaNet (linear attention)
            add_exl3(p + "attn_qkv", hl + "linear_attn.in_proj_qkvz");   // fused N=12288, sliced in fwd
            if (const auto* ba = need(hl + "linear_attn.in_proj_ba.weight"))  // F16 [64,2048]→ne=[2048,64]
                add_verbatim(p + "ssm_ba.weight", DType::kF16, {hidden, uint64_t(ba->shape[0])}, ba);
            add_exl3(p + "ssm_out", hl + "linear_attn.out_proj");
            add_negexp(p + "ssm_a",         {ssm_dt_rank}, need(hl + "linear_attn.A_log"));   // -exp(A_log); no .weight
            add_cast_f32(p + "ssm_dt.bias", {ssm_dt_rank}, need(hl + "linear_attn.dt_bias"));
            if (const auto* cv = need(hl + "linear_attn.conv1d.weight")) {  // BF16 [ch,1,k]→F32 ne=[k,ch]
                const uint64_t conv_ch = uint64_t(cv->numel()) / conv_kernel;
                add_cast_f32(p + "ssm_conv1d.weight", {conv_kernel, conv_ch}, cv);
            }
            add_cast_f32(p + "ssm_norm.weight", {ssm_state}, need(hl + "linear_attn.norm.weight"));
        } else {      // full attention (L%fa_interval == fa_interval-1)
            add_exl3(p + "attn_q",      hl + "self_attn.q_proj");   // fused q|gate (N=8192)
            add_exl3(p + "attn_k",      hl + "self_attn.k_proj");
            add_exl3(p + "attn_v",      hl + "self_attn.v_proj");
            add_exl3(p + "attn_output", hl + "self_attn.o_proj");
            add_norm(p + "attn_q_norm.weight", {head_dim}, need(hl + "self_attn.q_norm.weight"));
            add_norm(p + "attn_k_norm.weight", {head_dim}, need(hl + "self_attn.k_norm.weight"));
        }

        // MoE (every layer): router + fused experts + shared expert
        add_cast_f32(p + "ffn_gate_inp.weight", {hidden, n_expert}, need(hl + "mlp.gate.weight"));
        add_experts(L, "gate_proj", "ffn_gate_exps");
        add_experts(L, "up_proj",   "ffn_up_exps");
        add_experts(L, "down_proj", "ffn_down_exps");
        add_exl3(p + "ffn_gate_shexp", hl + "mlp.shared_expert.gate_proj");
        add_exl3(p + "ffn_up_shexp",   hl + "mlp.shared_expert.up_proj");
        add_exl3(p + "ffn_down_shexp", hl + "mlp.shared_expert.down_proj");
        add_cast_f32(p + "ffn_gate_inp_shexp.weight", {hidden},
                     need(hl + "mlp.shared_expert_gate.weight"));
    }
    if (!err.empty()) return err;

    // ---- pass 1: declare every tensor (writes header + offsets) ----
    for (const auto& o : outs) w.tensor_info(o.name, o.dt, o.shape, o.nbytes);

    // ---- pass 2: stream the bytes, one tensor / fused-bank in RAM at a time ----
    if (auto e = w.begin_streaming(out_gguf); !e.empty()) return "begin_streaming: " + e;
    std::vector<uint8_t> bank;       // reused scratch for fused banks (~335 MiB peak)
    std::vector<float>   cast;       // reused scratch for F32 casts
    int n_exl3 = 0, n_fused = 0, n_cast = 0, n_vb = 0;
    for (const auto& o : outs) {
        if (o.kind == Kind::Verbatim) {
            if (auto e = w.stream_next(o.src->data, o.nbytes); !e.empty()) return "stream: " + e;
            o.dt == DType::kEXL3 ? ++n_exl3 : ++n_vb;
        } else if (o.kind == Kind::CastF32 || o.kind == Kind::CastF32P1 || o.kind == Kind::CastNegExp) {
            cast = to_f32(o.src);
            if (o.kind == Kind::CastF32P1)      for (auto& x : cast) x += 1.0f;
            else if (o.kind == Kind::CastNegExp) for (auto& x : cast) x = -std::exp(x);
            if (auto e = w.stream_next(cast.data(), cast.size() * 4); !e.empty()) return "stream: " + e;
            ++n_cast;
        } else {  // Fused bank
            bank.resize(o.nbytes);
            uint64_t off = 0;
            for (const auto* s : o.srcs) { std::memcpy(bank.data() + off, s->data, s->nbytes); off += s->nbytes; }
            if (off != o.nbytes) return "fused size mismatch: " + o.name;
            if (auto e = w.stream_next(bank.data(), o.nbytes); !e.empty()) return "stream: " + e;
            ++n_fused;
        }
    }
    if (auto e = w.end_streaming(); !e.empty()) return "end_streaming: " + e;
    note("streamed " + std::to_string(outs.size()) + " tensors (" + std::to_string(n_exl3) +
         " EXL3, " + std::to_string(n_fused) + " fused-expert banks, " + std::to_string(n_cast) +
         " F32-cast, " + std::to_string(n_vb) + " verbatim)");
    note("wrote " + out_gguf);
    return {};
}

}  // namespace ie
