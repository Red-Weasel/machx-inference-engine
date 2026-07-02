// tools/inspect_gguf.cpp — Phase 1 gate tool.
//
//   ie-inspect <model.gguf>            list metadata + tensors with groups
//   ie-inspect <model.gguf> --kv       full KV dump
//   ie-inspect <model.gguf> --tensors  full tensor table only
//   ie-inspect <model.gguf> --layer N  show only blk.N.* tensors
//
// Phase 1 gate: parsing the unsloth Qwen3.6-35B-A3B-Q4_K_M.gguf must list
// every expected tensor with correct shape and dtype.

#include "ie/dtype.hpp"
#include "ie/gguf.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr const char* kReset = "\033[0m";
constexpr const char* kBold  = "\033[1m";
constexpr const char* kDim   = "\033[2m";
constexpr const char* kCyan  = "\033[36m";
constexpr const char* kGreen = "\033[32m";
constexpr const char* kYell  = "\033[33m";
constexpr const char* kRed   = "\033[31m";

void print_header(const ie::GgufReader& g) {
    std::printf("%sGGUF v%u%s  %llu tensors  %llu KV  align=%llu  data@%llu  size=%.2f GiB\n",
                kBold, g.version(), kReset,
                (unsigned long long)g.n_tensors(),
                (unsigned long long)g.n_kv(),
                (unsigned long long)g.alignment(),
                (unsigned long long)g.tensor_data_offset(),
                g.file_size() / double(1ull << 30));
}

void print_kv_one(const ie::KvRef& kv, bool truncate) {
    using V = ie::GgufValueType;
    std::printf("  %s%-46.*s%s  %s%-7.*s%s ",
                kCyan, int(kv.key.size()), kv.key.data(), kReset,
                kDim, int(ie::name_of(kv.type).size()), ie::name_of(kv.type).data(), kReset);
    switch (kv.type) {
        case V::kBool:   std::printf("%s\n", kv.as_bool() ? "true" : "false"); break;
        case V::kU8: case V::kU16: case V::kU32: case V::kU64:
            std::printf("%llu\n", (unsigned long long)kv.as_uint()); break;
        case V::kI8: case V::kI16: case V::kI32: case V::kI64:
            std::printf("%lld\n", (long long)kv.as_int()); break;
        case V::kF32: case V::kF64:
            std::printf("%g\n", kv.as_float()); break;
        case V::kString: {
            auto s = kv.as_string();
            if (truncate && s.size() > 80) {
                std::printf("\"%.*s...\" (%zu chars)\n",
                            80, s.data(), s.size());
            } else {
                std::printf("\"%.*s\"\n", int(s.size()), s.data());
            }
            break;
        }
        case V::kArray: {
            std::printf("[%llu × %s]", (unsigned long long)kv.n_array,
                        ie::name_of(kv.inner_type).data());
            if (kv.inner_type == V::kString && kv.n_array > 0 && truncate) {
                auto arr = kv.as_string_array();
                std::printf("  e.g.");
                for (size_t i = 0; i < std::min<size_t>(3, arr.size()); ++i) {
                    auto s = arr[i];
                    std::printf(" \"%.*s\"", int(std::min<size_t>(s.size(), 24)), s.data());
                    if (s.size() > 24) std::printf("...");
                }
            }
            std::putchar('\n');
            break;
        }
    }
}

// Highest-priority KV keys to surface even when --kv isn't given.
// Built dynamically from `general.architecture` (e.g. qwen35moe.*).
std::vector<std::string> pin_kvs(std::string_view arch) {
    std::vector<std::string> v = {
        "general.architecture",
        "general.name",
        "general.basename",
        "general.size_label",
        "general.file_type",
        "general.alignment",
        "general.quantization_version",
    };
    auto add_arch = [&](std::string_view suffix) {
        v.emplace_back(std::string(arch) + std::string(suffix));
    };
    if (!arch.empty()) {
        add_arch(".block_count");
        add_arch(".context_length");
        add_arch(".embedding_length");
        add_arch(".attention.head_count");
        add_arch(".attention.head_count_kv");
        add_arch(".attention.key_length");
        add_arch(".attention.value_length");
        add_arch(".attention.layer_norm_rms_epsilon");
        add_arch(".feed_forward_length");
        add_arch(".expert_count");
        add_arch(".expert_used_count");
        add_arch(".expert_feed_forward_length");
        add_arch(".expert_shared_feed_forward_length");
        add_arch(".rope.freq_base");
        add_arch(".rope.dimension_count");
        add_arch(".rope.scaling.type");
        add_arch(".full_attention_interval");
        add_arch(".ssm.state_size");
        add_arch(".ssm.conv_kernel");
        add_arch(".ssm.group_count");
        add_arch(".ssm.time_step_rank");
        add_arch(".ssm.inner_size");
        add_arch(".tie_lm_head");
    }
    v.insert(v.end(), {
        "tokenizer.ggml.model",
        "tokenizer.ggml.pre",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.padding_token_id",
        "tokenizer.ggml.add_bos_token",
    });
    return v;
}

void print_kv_pinned(const ie::GgufReader& g) {
    std::string arch;
    if (auto* a = g.find_kv("general.architecture")) arch = std::string(a->as_string());
    auto keys = pin_kvs(arch);
    std::printf("\n%sPinned metadata%s\n", kBold, kReset);
    for (auto& key : keys) {
        if (auto* kv = g.find_kv(key)) {
            print_kv_one(*kv, /*truncate=*/true);
        }
    }
}

void print_kv_full(const ie::GgufReader& g) {
    std::printf("\n%sAll %llu KV pairs%s\n", kBold,
                (unsigned long long)g.n_kv(), kReset);
    for (const auto& kv : g.kvs()) print_kv_one(kv, /*truncate=*/true);
}

void print_dtype_histogram(const ie::GgufReader& g) {
    std::map<ie::DType, std::pair<uint64_t, uint64_t>> hist; // count, bytes
    for (const auto& t : g.tensors()) {
        auto& e = hist[t.dtype];
        e.first  += 1;
        e.second += t.nbytes;
    }
    std::printf("\n%sDtype histogram%s\n", kBold, kReset);
    std::printf("  %-8s %10s %14s\n", "dtype", "tensors", "bytes (GiB)");
    for (auto& [d, e] : hist) {
        std::printf("  %-8.*s %10llu %14.3f\n",
                    int(ie::type_name(d).size()), ie::type_name(d).data(),
                    (unsigned long long)e.first,
                    e.second / double(1ull << 30));
    }
}

struct TensorGroup {
    std::string         label;
    int                 expected = -1;     // -1 = "no expectation"
    std::vector<const ie::GgufTensorInfo*> entries;
    bool                is_per_layer = false; // groups indexed by layer id
};

// Group key (sort order). The leading numeric prefix forces the desired
// rendering order in the std::map output.
static std::string keyf(int order, std::string suffix) {
    char buf[8]; std::snprintf(buf, sizeof(buf), "%02d_", order);
    return std::string(buf) + std::move(suffix);
}

// Bucket tensors. Expected counts assume the Qwen3.6-35B-A3B layer pattern
// (i%4 == 3 -> full-attn, others -> Gated-DeltaNet) when interval == 4.
// Other models will simply have grey "no expectation" lines.
std::map<std::string, TensorGroup> group_tensors(const ie::GgufReader& g, int n_layers_hint, int full_attn_interval) {
    std::map<std::string, TensorGroup> groups;
    auto bucket = [&](const std::string& key, std::string label, int expected) -> TensorGroup& {
        auto it = groups.find(key);
        if (it == groups.end()) {
            TensorGroup grp;
            grp.label = std::move(label);
            grp.expected = expected;
            it = groups.emplace(key, std::move(grp)).first;
        }
        return it->second;
    };

    int n_layers     = n_layers_hint;
    int n_full_attn  = -1;
    int n_linear     = -1;
    if (n_layers > 0 && full_attn_interval > 0) {
        n_full_attn = n_layers / full_attn_interval;
        n_linear    = n_layers - n_full_attn;
    }

    // Per-Qwen3.6 expectations (counts of distinct tensor names per layer):
    //   full-attn layer: 6 attn_* (q, k, v, output, q_norm, k_norm)
    //   DeltaNet layer:  9 (attn_gate, attn_qkv, ssm_a, ssm_alpha, ssm_beta,
    //                       ssm_conv1d, ssm_dt, ssm_norm, ssm_out)
    //   every layer:     2 norms (attn_norm, post_attention_norm)
    //   every layer:     8 ffn (gate_inp, gate_inp_shexp, gate_exps, up_exps,
    //                            down_exps, gate_shexp, up_shexp, down_shexp)
    const int per_full_attn = 6;
    const int per_delta     = 9;
    const int per_norms     = 2;
    const int per_ffn       = 8;

    static const std::regex re_blk(R"(^blk\.(\d+)\.(.+)$)");

    for (const auto& t : g.tensors()) {
        std::string name(t.name);
        std::smatch m;
        if (std::regex_match(name, m, re_blk)) {
            int idx = std::atoi(m[1].str().c_str());
            std::string sub = m[2].str();
            const bool is_full = (full_attn_interval > 0) && (idx % full_attn_interval == full_attn_interval - 1);

            const bool is_norm = (sub == "attn_norm.weight" || sub == "post_attention_norm.weight");
            const bool is_ffn  = sub.rfind("ffn_", 0) == 0;
            const bool is_ssm  = sub.rfind("ssm_", 0) == 0;
            const bool is_attn = sub.rfind("attn_", 0) == 0;

            if (is_norm) {
                bucket(keyf(20, "layer_norms"),
                       "blk.{i}.{attn_norm,post_attention_norm}.weight",
                       n_layers > 0 ? n_layers * per_norms : -1).entries.push_back(&t);
            } else if (is_ffn) {
                bucket(keyf(50, "ffn_moe"),
                       "blk.{i}.ffn_* (router + 256 experts + 1 shared)",
                       n_layers > 0 ? n_layers * per_ffn : -1).entries.push_back(&t);
            } else if (is_full && is_attn) {
                bucket(keyf(30, "full_attn"),
                       "blk.{i}.attn_* (full-attn, i%4==3)",
                       n_full_attn > 0 ? n_full_attn * per_full_attn : -1).entries.push_back(&t);
            } else if (!is_full && (is_ssm || sub == "attn_qkv.weight" || sub == "attn_gate.weight")) {
                bucket(keyf(40, "delta_net"),
                       "blk.{i}.{attn_qkv,attn_gate,ssm_*} (DeltaNet, i%4!=3)",
                       n_linear > 0 ? n_linear * per_delta : -1).entries.push_back(&t);
            } else {
                bucket(keyf(98, "layers_unrecognized"),
                       "blk.{i}.<unrecognized>", -1).entries.push_back(&t);
            }
        } else if (name == "token_embd.weight") {
            bucket(keyf(0, "embed"), "token_embd.weight", 1).entries.push_back(&t);
        } else if (name == "output.weight") {
            bucket(keyf(1, "lm_head"), "output.weight (lm_head)", 1).entries.push_back(&t);
        } else if (name == "output_norm.weight") {
            bucket(keyf(2, "final_norm"), "output_norm.weight", 1).entries.push_back(&t);
        } else if (name.find("mtp") != std::string::npos) {
            bucket(keyf(60, "mtp"), "mtp.* (multi-token-prediction head)", -1).entries.push_back(&t);
        } else if (name.find("v.") == 0 || name.find("vision") != std::string::npos) {
            bucket(keyf(70, "vision"), "vision_tower.* (skip if text-only)", -1).entries.push_back(&t);
        } else {
            bucket(keyf(99, "other"), "other top-level", -1).entries.push_back(&t);
        }
    }
    return groups;
}

void print_tensor(const ie::GgufTensorInfo& t) {
    char shape_buf[80];
    int p = 0;
    p += std::snprintf(shape_buf + p, sizeof(shape_buf) - p, "[");
    for (uint32_t i = 0; i < t.n_dims; ++i) {
        if (i) p += std::snprintf(shape_buf + p, sizeof(shape_buf) - p, ", ");
        p += std::snprintf(shape_buf + p, sizeof(shape_buf) - p, "%llu",
                           (unsigned long long)t.shape[i]);
    }
    p += std::snprintf(shape_buf + p, sizeof(shape_buf) - p, "]");
    std::printf("    %s%-6.*s%s %-32s %.3f MiB  %.*s\n",
                kCyan, int(ie::type_name(t.dtype).size()), ie::type_name(t.dtype).data(), kReset,
                shape_buf, t.nbytes / double(1ull << 20),
                int(t.name.size()), t.name.data());
}

void print_groups(const std::map<std::string, TensorGroup>& groups, bool verbose) {
    std::printf("\n%sTensor groups%s\n", kBold, kReset);
    for (const auto& [k, grp] : groups) {
        size_t actual = grp.entries.size();
        std::string status;
        if (grp.expected < 0) {
            status = std::string(kDim) + "—" + kReset;
        } else if (int(actual) == grp.expected) {
            status = std::string(kGreen) + "OK" + kReset;
        } else {
            status = std::string(kRed) + "MISMATCH" + kReset;
        }
        std::printf("  [%s] %-50s %4zu found", status.c_str(), grp.label.c_str(), actual);
        if (grp.expected >= 0) std::printf("  (expect %d)", grp.expected);
        std::putchar('\n');

        if (verbose) {
            // Sort entries by name for stable output
            auto sorted = grp.entries;
            std::sort(sorted.begin(), sorted.end(),
                      [](auto* a, auto* b) { return a->name < b->name; });
            for (auto* t : sorted) print_tensor(*t);
        }
    }
}

void print_tensors_full(const ie::GgufReader& g) {
    auto v = g.tensors();
    std::vector<const ie::GgufTensorInfo*> sorted;
    sorted.reserve(v.size());
    for (auto& t : v) sorted.push_back(&t);
    std::sort(sorted.begin(), sorted.end(),
              [](auto* a, auto* b) { return a->name < b->name; });
    std::printf("\n%sAll %zu tensors%s\n", kBold, v.size(), kReset);
    for (auto* t : sorted) print_tensor(*t);
}

void print_layer_filter(const ie::GgufReader& g, int layer) {
    char prefix[32];
    std::snprintf(prefix, sizeof(prefix), "blk.%d.", layer);
    std::printf("\n%sLayer %d (prefix '%s')%s\n", kBold, layer, prefix, kReset);
    bool any = false;
    std::vector<const ie::GgufTensorInfo*> ts;
    for (const auto& t : g.tensors()) {
        if (std::string(t.name).rfind(prefix, 0) == 0) ts.push_back(&t);
    }
    std::sort(ts.begin(), ts.end(), [](auto* a, auto* b) { return a->name < b->name; });
    for (auto* t : ts) { print_tensor(*t); any = true; }
    if (!any) std::printf("  %s(no tensors with prefix %s)%s\n", kYell, prefix, kReset);
}

void usage() {
    std::fputs(
        "Usage: ie-inspect <model.gguf> [--kv] [--tensors] [--layer N]\n"
        "  default       summary: header, pinned metadata, dtype histogram, tensor groups\n"
        "  --kv          dump every KV pair\n"
        "  --tensors     dump every tensor (sorted by name)\n"
        "  --layer N     dump only tensors with prefix 'blk.N.'\n",
        stderr);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 2; }

    bool dump_kv = false, dump_tensors = false;
    int layer_filter = -1;
    std::string path;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--kv") dump_kv = true;
        else if (a == "--tensors") dump_tensors = true;
        else if (a == "--layer" && i+1 < argc) layer_filter = std::atoi(argv[++i]);
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (path.empty()) path = a;
        else { usage(); return 2; }
    }
    if (path.empty()) { usage(); return 2; }

    ie::GgufReader g;
    if (auto err = g.open(path); !err.empty()) {
        std::fprintf(stderr, "%s%s%s\n", kRed, err.c_str(), kReset);
        return 1;
    }

    print_header(g);
    print_kv_pinned(g);
    if (dump_kv) print_kv_full(g);
    print_dtype_histogram(g);

    // Discover model dims from the architecture-prefixed KVs.
    int n_layers = 0;
    int full_attn_interval = 0;
    if (auto* a = g.find_kv("general.architecture"); a) {
        std::string arch(a->as_string());
        if (auto* kv = g.find_kv(arch + ".block_count")) n_layers = int(kv->as_uint());
        if (auto* kv = g.find_kv(arch + ".full_attention_interval")) full_attn_interval = int(kv->as_uint());
    }

    auto groups = group_tensors(g, n_layers, full_attn_interval);
    // Total cross-check
    size_t total_actual = 0;
    int total_expected = 0;
    bool any_unknown = false;
    for (auto& [k, grp] : groups) {
        total_actual += grp.entries.size();
        if (grp.expected < 0) any_unknown = true;
        else total_expected += grp.expected;
    }
    print_groups(groups, /*verbose=*/dump_tensors && layer_filter < 0);

    std::printf("\n%sTotal%s  %zu tensors found", kBold, kReset, total_actual);
    if (!any_unknown) std::printf("  vs %d expected — %s",
                                  total_expected,
                                  int(total_actual) == total_expected
                                      ? (std::string(kGreen) + "GATE PASS" + kReset).c_str()
                                      : (std::string(kRed) + "GATE FAIL" + kReset).c_str());
    std::putchar('\n');

    if (layer_filter >= 0) print_layer_filter(g, layer_filter);
    if (dump_tensors && layer_filter < 0) print_tensors_full(g);

    return 0;
}
