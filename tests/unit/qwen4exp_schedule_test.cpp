// tests/unit/qwen4exp_schedule_test.cpp -- read_qwen4exp_config on synthetic GGUF headers (Flash-Next's values, written with
// GgufWriter): what llama.cpp's hparams reading accepts and the engine must too on Qwen 4 day one (2026-11-07) -- the layer
// schedule as a per-block attention.recurrent_layers array (BOOL or I32 elements) or the interval rule with the key defaulting
// to 4; nextn_predict_layers (MTP blocks counted in block_count) taken off n_layers; head counts / expert_feed_forward_length as
// uniform per-block arrays; and the explicit limits where the kernels are fixed (the QSA block of 4 cells: compress ratio,
// top_k). Host-only, no GPU, no model file.
#undef NDEBUG
#include "ie/gguf.hpp"
#include "ie/gguf_writer.hpp"
#include "ie/model_config.hpp"

#include <cstdio>
#include <functional>
#include <string>
#include <unistd.h>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) { std::printf("[%s] %s\n", ok ? " ok " : "FAIL", what.c_str()); if (!ok) ++g_fail; }

struct Params {
    uint32_t block_count = 48;
    uint32_t nextn = 0;                      // written when > 0
    bool interval_key = true;                // full_attention_interval 4
    std::vector<bool> recurrent;             // attention.recurrent_layers as BOOL when non-empty
    std::vector<int32_t> recurrent_i32;      // ... as I32 when non-empty
    std::vector<int32_t> compress;           // attention.compress_ratios when non-empty
    std::vector<int32_t> ffn_arr;            // expert_feed_forward_length as an array when non-empty (else scalar 640)
    std::vector<int32_t> kv_heads_arr;       // attention.head_count_kv as an array when non-empty (else scalar 2)
    uint32_t top_k = 2048;
};
// Flash-Next's header values (tools/qwen4exp_config_test.cpp on the daily file); no PLE group (optional, all-or-none).
std::string write_q4e(const std::string& path, const Params& p) {
    ie::GgufWriter w;
    w.kv_string("general.architecture", "qwen4exp");
    w.kv_u32("qwen4exp.block_count", p.block_count);
    if (p.nextn) w.kv_u32("qwen4exp.nextn_predict_layers", p.nextn);
    w.kv_u32("qwen4exp.embedding_length", 2560);
    w.kv_u32("qwen4exp.context_length", 262144);
    w.kv_u32("qwen4exp.attention.head_count", 24);
    if (p.kv_heads_arr.empty()) w.kv_u32("qwen4exp.attention.head_count_kv", 2);
    else w.kv_i32_array("qwen4exp.attention.head_count_kv", p.kv_heads_arr);
    w.kv_u32("qwen4exp.attention.key_length", 256);
    w.kv_u32("qwen4exp.attention.value_length", 256);
    if (p.interval_key) w.kv_u32("qwen4exp.full_attention_interval", 4);
    if (!p.recurrent.empty()) w.kv_bool_array("qwen4exp.attention.recurrent_layers", p.recurrent);
    if (!p.recurrent_i32.empty()) w.kv_i32_array("qwen4exp.attention.recurrent_layers", p.recurrent_i32);
    w.kv_f32("qwen4exp.attention.layer_norm_rms_epsilon", 1e-6f);
    w.kv_u32("qwen4exp.rope.dimension_count", 64);
    w.kv_f32("qwen4exp.rope.freq_base", 1e7f);
    w.kv_i32_array("qwen4exp.rope.dimension_sections", {11, 11, 10, 0});
    w.kv_u32("qwen4exp.ssm.state_size", 128);
    w.kv_u32("qwen4exp.ssm.conv_kernel", 4);
    w.kv_u32("qwen4exp.ssm.group_count", 16);
    w.kv_u32("qwen4exp.ssm.time_step_rank", 48);
    w.kv_u32("qwen4exp.ssm.inner_size", 6144);
    w.kv_u32("qwen4exp.hyper_connection.count", 4);
    w.kv_u32("qwen4exp.hyper_connection.low_rank", 320);
    w.kv_u32("qwen4exp.attention.indexer.head_count", 4);
    w.kv_u32("qwen4exp.attention.indexer.key_length", 128);
    w.kv_u32("qwen4exp.attention.indexer.top_k", p.top_k);
    if (!p.compress.empty()) w.kv_i32_array("qwen4exp.attention.compress_ratios", p.compress);
    w.kv_u32("qwen4exp.expert_count", 512);
    w.kv_u32("qwen4exp.expert_used_count", 10);
    if (p.ffn_arr.empty()) w.kv_u32("qwen4exp.expert_feed_forward_length", 640);
    else w.kv_i32_array("qwen4exp.expert_feed_forward_length", p.ffn_arr);
    w.kv_u32("qwen4exp.expert_shared_feed_forward_length", 640);
    w.kv_string_array("tokenizer.ggml.tokens", {"a", "b", "c"});
    return w.write(path);
}
// Reads a synthetic file; returns the reader's message ("" = accepted) and fills c.
std::string read(const Params& p, ie::Qwen4ExpConfig& c) {
    const std::string path = "/tmp/qwen4exp_schedule_test_" + std::to_string(getpid()) + ".gguf";
    if (auto e = write_q4e(path, p); !e.empty()) return "write: " + e;
    ie::GgufReader g;
    if (auto e = g.open(path); !e.empty()) return "open: " + e;
    const std::string r = ie::read_qwen4exp_config(g, c);
    ::unlink(path.c_str());
    return r;
}
bool has(const std::string& s, const char* sub) { return s.find(sub) != std::string::npos; }
std::vector<int32_t> ratios(uint32_t n, const std::function<bool(uint32_t)>& full, int32_t r = 4) {
    std::vector<int32_t> v(n, 0);
    for (uint32_t il = 0; il < n; ++il) v[il] = full(il) ? r : 0;
    return v;
}
}  // namespace

int main() {
    auto every4th = [](uint32_t il) { return il % 4 == 3; };
    {   // Flash-Next as shipped
        Params p; p.compress = ratios(48, every4th);
        ie::Qwen4ExpConfig c; const std::string e = read(p, c);
        check(e.empty(), "the shipped header is accepted: " + e);
        check(c.n_layers == 48 && c.nextn_predict_layers == 0 && c.full_attention_interval == 4, "48 trunk blocks, no nextn, interval 4");
        check(c.n_full_attn_layers() == 12 && c.n_linear_layers() == 36, "12 full-attention + 36 linear blocks");
        check(c.is_full_attn(3) && c.is_full_attn(47) && !c.is_full_attn(0) && !c.is_full_attn(46), "the interval rule: 3, 47 yes; 0, 46 no");
        check(c.recurrent_layers.empty(), "no recurrent_layers array recorded");
    }
    {   // llama.cpp's nextn export inside the main file: block_count 49, one MTP block, the schedule arrays 49 long
        Params p; p.block_count = 49; p.nextn = 1; p.compress = ratios(49, every4th);
        ie::Qwen4ExpConfig c; const std::string e = read(p, c);
        check(e.empty(), "block_count 49 with nextn_predict_layers 1 is accepted: " + e);
        check(c.n_layers == 48 && c.nextn_predict_layers == 1 && c.n_full_attn_layers() == 12, "n_layers is the trunk (48), nextn 1 recorded");
    }
    {   // nextn >= block_count
        Params p; p.block_count = 48; p.nextn = 48;
        ie::Qwen4ExpConfig c; const std::string e = read(p, c);
        check(has(e, "nextn_predict_layers >= block_count"), "nextn_predict_layers >= block_count is refused: " + e);
    }
    {   // full_attention_interval absent, no array: llama.cpp's default 4
        Params p; p.interval_key = false; p.compress = ratios(48, every4th);
        ie::Qwen4ExpConfig c; const std::string e = read(p, c);
        check(e.empty() && c.full_attention_interval == 4 && c.n_full_attn_layers() == 12, "a missing interval key defaults to 4: " + e);
    }
    {   // attention.recurrent_layers (BOOL) spelling the same schedule, interval key present
        Params p; p.compress = ratios(48, every4th);
        for (uint32_t il = 0; il < 48; ++il) p.recurrent.push_back(!every4th(il));
        ie::Qwen4ExpConfig c; const std::string e = read(p, c);
        check(e.empty() && c.recurrent_layers.size() == 48 && c.n_full_attn_layers() == 12 && c.is_full_attn(3) && !c.is_full_attn(0),
              "a BOOL recurrent_layers array with the regular schedule: " + e);
    }
    {   // an irregular schedule only the array can express, no interval key, compress ratios agreeing with it
        auto full = [](uint32_t il) { return il == 5 || il == 11 || il == 23 || il == 47; };
        Params p; p.interval_key = false; p.compress = ratios(48, full);
        for (uint32_t il = 0; il < 48; ++il) p.recurrent.push_back(!full(il));
        ie::Qwen4ExpConfig c; const std::string e = read(p, c);
        check(e.empty(), "an irregular recurrent_layers schedule is accepted: " + e);
        check(c.n_full_attn_layers() == 4 && c.is_full_attn(5) && c.is_full_attn(11) && !c.is_full_attn(3) && !c.is_full_attn(46),
              "is_full_attn follows the array (5, 11 yes; 3, 46 no), 4 full-attention blocks");
    }
    {   // the array as I32 elements
        Params p; p.compress = ratios(48, every4th);
        for (uint32_t il = 0; il < 48; ++il) p.recurrent_i32.push_back(every4th(il) ? 0 : 1);
        ie::Qwen4ExpConfig c; const std::string e = read(p, c);
        check(e.empty() && c.recurrent_layers.size() == 48 && c.n_full_attn_layers() == 12, "an I32 recurrent_layers array: " + e);
    }
    {   // the array 49 long (trunk + the nextn entry) on a 49-block file: the nextn entry is dropped
        Params p; p.block_count = 49; p.nextn = 1; p.compress = ratios(49, every4th);
        for (uint32_t il = 0; il < 49; ++il) p.recurrent.push_back(!every4th(il));
        ie::Qwen4ExpConfig c; const std::string e = read(p, c);
        check(e.empty() && c.recurrent_layers.size() == 48 && c.n_full_attn_layers() == 12, "a 49-entry array on a 49-block file keeps the 48 trunk entries: " + e);
    }
    {   // the array of a wrong length
        Params p; p.compress = ratios(48, every4th);
        for (uint32_t il = 0; il < 40; ++il) p.recurrent.push_back(!every4th(il));
        ie::Qwen4ExpConfig c; const std::string e = read(p, c);
        check(has(e, "recurrent_layers length"), "a 40-entry array on a 48-block file is refused: " + e);
    }
    {   // compress ratios disagreeing with the schedule (existing rule) and a ratio the kernels do not pool
        Params p; p.compress = ratios(48, [](uint32_t il) { return il % 4 == 2; });
        ie::Qwen4ExpConfig c; std::string e = read(p, c);
        check(has(e, "disagrees"), "compress ratios on the wrong blocks are refused: " + e);
        Params p8; p8.compress = ratios(48, every4th, 8);
        e = read(p8, c);
        check(has(e, "blocks of 4"), "compress ratio 8 names the kernels' 4-cell block: " + e);
    }
    {   // top_k not a multiple of 4
        Params p; p.compress = ratios(48, every4th); p.top_k = 2050;
        ie::Qwen4ExpConfig c; const std::string e = read(p, c);
        check(has(e, "multiple of the 4-cell"), "indexer top_k 2050 is refused: " + e);
    }
    {   // expert_feed_forward_length as a uniform per-block array, and as a differing one
        Params p; p.compress = ratios(48, every4th); p.ffn_arr.assign(48, 640);
        ie::Qwen4ExpConfig c; std::string e = read(p, c);
        check(e.empty() && c.expert_ffn == 640, "a uniform expert_feed_forward_length array is the scalar: " + e);
        p.ffn_arr[7] = 1280;
        e = read(p, c);
        check(has(e, "differs per block"), "a differing expert_feed_forward_length array is refused: " + e);
    }
    {   // attention.head_count_kv as a uniform array
        Params p; p.compress = ratios(48, every4th); p.kv_heads_arr.assign(48, 2);
        ie::Qwen4ExpConfig c; const std::string e = read(p, c);
        check(e.empty() && c.n_kv_heads == 2, "a uniform head_count_kv array is the scalar: " + e);
    }
    std::printf("qwen4exp_schedule_test: %s (%d failures)\n", g_fail ? "FAILED" : "OK", g_fail);
    return g_fail ? 1 : 0;
}
