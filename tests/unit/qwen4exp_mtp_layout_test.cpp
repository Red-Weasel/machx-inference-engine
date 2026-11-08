// tests/unit/qwen4exp_mtp_layout_test.cpp -- the MTP draft head's two file layouts (include/ie/qwen4exp.hpp:
// qwen4exp_mtp_nextn_block, qwen4exp_mtp_tensor_name, qwen4exp_mtp_split_eh_proj). Host-only, no GPU. The synthetic checks
// always run. With a llama.cpp nextn export (argv[1] or IE_QWEN35_MTP_GGUF, e.g. unsloth's
// MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf): the nextn block is block_count - nextn_predict_layers, every tensor load_mtp
// binds resolves to a tensor of that file, eh_proj is [2*hidden, hidden], and read_qwen4exp_config reports the trunk count.
// With an engine-layout head as argv[2] (scripts/mtp_to_gguf.py's mtp-head.gguf): it is NOT taken for a nextn export.
#include "ie/model_config.hpp"
#include "ie/qwen4exp.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) { std::printf("[%s] %s\n", ok ? " ok " : "FAIL", what.c_str()); if (!ok) ++g_fail; }
// every engine-layout name Qwen4ExpModel::load_mtp binds from a nextn export (the fusion as the fused eh_proj)
const char* const kBound[] = {
    "mtp.pre_fc_norm_hidden.weight", "mtp.pre_fc_norm_embedding.weight", "mtp.eh_proj.weight",
    "mtp.attn_q.weight", "mtp.attn_k.weight", "mtp.attn_v.weight", "mtp.attn_output.weight",
    "mtp.attn_q_norm.weight", "mtp.attn_k_norm.weight",
    "mtp.indexer.q_proj.weight", "mtp.indexer.k_proj.weight", "mtp.indexer.q_norm.weight", "mtp.indexer.k_norm.weight",
    "mtp.hc_attn_norm.weight", "mtp.hc_attn_down.weight", "mtp.hc_attn_up.weight", "mtp.hc_attn_inject.weight",
    "mtp.hc_ffn_norm.weight", "mtp.hc_ffn_down.weight", "mtp.hc_ffn_up.weight", "mtp.hc_ffn_inject.weight",
    "mtp.ffn_gate_inp.weight", "mtp.ffn_gate_inp_shexp.weight",
    "mtp.ffn_gate_shexp.weight", "mtp.ffn_up_shexp.weight", "mtp.ffn_down_shexp.weight",
    "mtp.ffn_gate_exps.weight", "mtp.ffn_up_exps.weight", "mtp.ffn_down_exps.weight",
    "mtp.out_hc_norm.weight", "mtp.out_hc_down.weight", "mtp.out_hc_up.weight"};
}  // namespace

int main(int argc, char** argv) {
    using ie::qwen4exp_mtp_tensor_name;
    // -- the name mapping --------------------------------------------------------------------------------------------------
    check(qwen4exp_mtp_tensor_name("mtp.attn_q.weight", -1) == "mtp.attn_q.weight", "layout (a): a name is unchanged");
    check(qwen4exp_mtp_tensor_name("mtp.pre_fc_norm_hidden.weight", -1) == "mtp.pre_fc_norm_hidden.weight", "layout (a): the fusion norm too");
    check(qwen4exp_mtp_tensor_name("mtp.attn_q.weight", 48) == "blk.48.attn_q.weight", "layout (b): the block's own tensor");
    check(qwen4exp_mtp_tensor_name("mtp.indexer.q_proj.weight", 48) == "blk.48.indexer.q_proj.weight", "layout (b): a dotted sub-name");
    check(qwen4exp_mtp_tensor_name("mtp.hc_ffn_inject.weight", 48) == "blk.48.hc_ffn_inject.weight", "layout (b): hyper-connection");
    check(qwen4exp_mtp_tensor_name("mtp.ffn_gate_exps.weight", 48) == "blk.48.ffn_gate_exps.weight", "layout (b): an expert bank");
    check(qwen4exp_mtp_tensor_name("mtp.pre_fc_norm_embedding.weight", 48) == "blk.48.nextn.enorm.weight", "layout (b): pre_fc_norm_embedding = nextn.enorm");
    check(qwen4exp_mtp_tensor_name("mtp.pre_fc_norm_hidden.weight", 48) == "blk.48.nextn.hnorm.weight", "layout (b): pre_fc_norm_hidden = nextn.hnorm");
    check(qwen4exp_mtp_tensor_name("mtp.eh_proj.weight", 48) == "blk.48.nextn.eh_proj.weight", "layout (b): the fused projection");
    check(qwen4exp_mtp_tensor_name("mtp.out_hc_norm.weight", 48) == "blk.48.nextn.hc_head_norm.weight", "layout (b): out_hc_norm = nextn.hc_head_norm");
    check(qwen4exp_mtp_tensor_name("mtp.out_hc_down.weight", 48) == "blk.48.nextn.hc_head_down.weight", "layout (b): out_hc_down = nextn.hc_head_down");
    check(qwen4exp_mtp_tensor_name("mtp.out_hc_up.weight", 48) == "blk.48.nextn.hc_head_up.weight", "layout (b): out_hc_up = nextn.hc_head_up");
    check(qwen4exp_mtp_tensor_name("mtp.attn_q.weight", 92) == "blk.92.attn_q.weight", "layout (b): any block index");
    check(qwen4exp_mtp_tensor_name("token_embd.weight", 48) == "token_embd.weight", "a name outside mtp.* is unchanged");

    // -- the eh_proj split: [H rows][2H] -> fc_embed [K, N] from the first H columns, fc_hidden from the last H ----------
    {
        const uint32_t H = 4;
        std::vector<float> eh(size_t(H) * 2 * H);
        for (uint32_t n = 0; n < H; ++n)
            for (uint32_t k = 0; k < 2 * H; ++k) eh[size_t(n) * 2 * H + k] = float(n * 100 + k);
        std::vector<sycl::half> fe(size_t(H) * H), fh(size_t(H) * H);
        ie::qwen4exp_mtp_split_eh_proj(eh.data(), H, fe.data(), fh.data());
        bool ok = true;
        for (uint32_t k = 0; k < H; ++k)
            for (uint32_t n = 0; n < H; ++n) {
                ok = ok && float(fe[size_t(k) * H + n]) == float(n * 100 + k);
                ok = ok && float(fh[size_t(k) * H + n]) == float(n * 100 + H + k);
            }
        check(ok, "eh_proj [H][2H]: columns [0,H) -> fc_embed [K,N], columns [H,2H) -> fc_hidden [K,N]");
    }

    // -- a real nextn export ------------------------------------------------------------------------------------------------
    const char* path = argc > 1 ? argv[1] : std::getenv("IE_QWEN35_MTP_GGUF");
    if (path && *path) {
        ie::GgufReader g;
        if (auto e = g.open(path); !e.empty()) check(false, std::string("open ") + path + ": " + e);
        else {
            const int b = ie::qwen4exp_mtp_nextn_block(g);
            const ie::KvRef* bc = g.find_kv("qwen4exp.block_count");
            const ie::KvRef* nn = g.find_kv("qwen4exp.nextn_predict_layers");
            const ie::KvRef* hid = g.find_kv("qwen4exp.embedding_length");
            check(b >= 0 && bc && nn && uint64_t(b) == bc->as_uint() - nn->as_uint(),
                  "nextn export: the draft block is block_count - nextn_predict_layers (" + std::to_string(b) + ")");
            for (const char* n : kBound) {
                const std::string name = qwen4exp_mtp_tensor_name(n, b);
                check(g.find_tensor(name) != nullptr, std::string(n) + " -> " + name + " present");
            }
            const ie::GgufTensorInfo* eh = g.find_tensor(qwen4exp_mtp_tensor_name("mtp.eh_proj.weight", b));
            check(eh && hid && eh->n_dims == 2 && eh->shape[0] == 2 * hid->as_uint() && eh->shape[1] == hid->as_uint(),
                  "nextn.eh_proj is [2*hidden, hidden]");
            ie::Qwen4ExpConfig c;
            if (auto e = ie::read_qwen4exp_config(g, c); e.empty())
                check(c.n_layers == bc->as_uint() - nn->as_uint() && c.nextn_predict_layers == nn->as_uint(),
                      "read_qwen4exp_config: n_layers is the trunk count, nextn_predict_layers read");
            else std::printf("  [skip] read_qwen4exp_config on the head file: %s (an MTP-only file may lack trunk-only keys)\n", e.c_str());
        }
    } else std::printf("  (no nextn export given: the real-file checks are skipped; pass a path or set IE_QWEN35_MTP_GGUF)\n");
    if (argc > 2) {
        ie::GgufReader g;
        if (auto e = g.open(argv[2]); !e.empty()) check(false, std::string("open ") + argv[2] + ": " + e);
        else check(ie::qwen4exp_mtp_nextn_block(g) < 0 && g.find_tensor("mtp.pre_fc_norm_hidden.weight") != nullptr,
                   "an engine-layout head is not taken for a nextn export");
    }
    std::printf("qwen4exp_mtp_layout_test: %s (%d failures)\n", g_fail ? "FAILED" : "OK", g_fail);
    return g_fail ? 1 : 0;
}
