// tests/unit/mimo26_model_test.cpp — MiMo-V2.6 P1 gate (docs/mimo26/00_PORT_PLAN.md): the config
// parses, every text-path tensor binds with the expected shape/dtype and nothing is left unclaimed,
// and the host dequant references equal the independent values tools/mimo26/ref_dump.py wrote from
// llama.cpp's converter (qkv, TP-aware) and gguf-py (experts). Host-only; needs the checkpoint and
// the dump: SKIPs (77) without either, never passes vacuously.
#undef NDEBUG  // Release build (-DNDEBUG); asserts must stay live
#include "ie/mimo26.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace ie;

namespace {

std::vector<float> read_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<float> v;
    if (!f) return v;
    f.seekg(0, std::ios::end);
    const auto n = size_t(f.tellg()) / 4;
    f.seekg(0);
    v.resize(n);
    f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * 4));
    return v;
}

// Exact comparison (-0 == +0; a NaN anywhere is a failure). Prints the first mismatches.
bool same(const char* tag, const std::vector<float>& ref, const float* got, size_t n, size_t stride = 1) {
    if (ref.size() != (n + stride - 1) / stride) {
        std::printf("  %-28s FAIL: reference has %zu values, engine %zu\n", tag, ref.size(), (n + stride - 1) / stride);
        return false;
    }
    size_t bad = 0;
    for (size_t i = 0, j = 0; i < n; i += stride, ++j) {
        const float a = ref[j], b = got[i];
        if (a != b || std::isnan(a) || std::isnan(b)) {
            if (bad < 3) std::printf("  %-28s mismatch at %zu: ref %.9g engine %.9g\n", tag, i, double(a), double(b));
            ++bad;
        }
    }
    std::printf("  %-28s %s (%zu values%s)\n", tag, bad ? "FAIL" : "ok", ref.size(), stride > 1 ? ", sampled" : "");
    return bad == 0;
}

}  // namespace

int main() {
    const char* d = std::getenv("IE_MIMO26_DIR");
    const std::string dir = d && *d ? std::string(d) : std::string(std::getenv("HOME")) + "/models/MiMo-V2.6-Flash-RL";
    if (!fs::exists(dir + "/config.json")) { std::printf("SKIP: no checkpoint at %s\n", dir.c_str()); return 77; }

    // ---- 1. config ---------------------------------------------------------------------------
    Mimo26Config c;
    {
        std::ifstream f(dir + "/config.json", std::ios::binary);
        const std::string body((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        const std::string e = mimo26_parse_config(body, c);
        if (!e.empty()) std::printf("config: %s\n", e.c_str());
        assert(e.empty());
    }
    assert(c.n_layers == 48 && c.n_mtp_layers == 3 && c.dim == 4096 && c.inter_dim == 16384 && c.moe_inter_dim == 2048);
    assert(c.n_heads == 64 && c.n_kv_heads == 4 && c.n_kv_heads_swa == 8 && c.head_dim == 192 && c.v_head_dim == 128);
    assert(c.window == 128 && c.rope_dim() == 64 && c.n_routed_experts == 256 && c.n_activated_experts == 8);
    assert(c.fp8_block_n == 128 && c.fp8_block_k == 128 && c.mxfp4_block == 32 && c.norm_topk_prob && c.route_scale == 1.f);
    assert(std::fabs(c.value_scale - 0.707f) < 1e-6f && std::fabs(c.norm_eps - 1e-6f) < 1e-12f);
    assert(c.rope_theta == 1e7f && c.rope_theta_swa == 1e4f && c.eos_id == 151645 && c.pad_id == 151643);
    assert(!c.is_swa(0) && c.is_swa(1) && !c.is_swa(5) && c.is_swa(47) == false && c.is_swa(48));   // 47 is a full layer
    assert(!c.is_moe(0) && c.is_moe(1) && c.is_moe(47) && !c.is_moe(48));
    assert(!c.has_sink(0) && c.has_sink(1) && c.has_sink(48) && c.n_kv(0) == 4 && c.n_kv(1) == 8 && c.n_kv(48) == 8);
    assert(c.qkv_rows(0) == 13568 && c.qkv_rows(1) == 14848);
    std::printf("config ok\n");

    // ---- 2. bind -----------------------------------------------------------------------------
    Mimo26Model m;
    {
        const std::string e = m.load(dir);
        if (!e.empty()) std::printf("load: %s\n", e.c_str());
        assert(e.empty());
    }
    if (!m.unclaimed().empty()) {
        std::printf("unclaimed: %zu, e.g.", m.unclaimed().size());
        for (size_t i = 0; i < m.unclaimed().size() && i < 5; ++i) std::printf(" %s", m.unclaimed()[i].c_str());
        std::printf("\n");
    }
    assert(m.unclaimed().empty());
    assert(m.layers().size() == 51);
    {
        const auto& L0 = m.layers()[0];
        assert(!L0.swa && !L0.moe && !L0.mtp && !L0.has_sink && L0.n_kv == 4 && L0.qkv_tp == 4);
        assert(L0.exp_w1.empty() && L0.mlp_gate && L0.mlp_up && L0.mlp_down && !L0.gate_w && !L0.sink);
        const auto& L1 = m.layers()[1];
        assert(L1.swa && L1.moe && L1.has_sink && L1.n_kv == 8 && L1.qkv_tp == 4);
        assert(L1.exp_w1.size() == 256 && L1.exp_w3.size() == 256 && L1.exp_w2.size() == 256 && L1.gate_w && L1.gate_bias && L1.sink && !L1.mlp_gate);
        assert(L1.exp_w1[255].w && L1.exp_w1[255].s && L1.exp_w2[0].w->shape[0] == 4096 && L1.exp_w2[0].s->shape[1] == 64);
        const auto& L5 = m.layers()[5];
        assert(!L5.swa && L5.moe && !L5.has_sink && L5.n_kv == 4);
        const auto& M0 = m.layers()[48];
        assert(M0.mtp && M0.swa && M0.has_sink && !M0.moe && M0.eh_proj && M0.enorm && M0.hnorm && M0.final_norm && M0.mlp_gate);
        assert(m.embed && m.lm_head && m.final_norm);
    }
    // 47 MoE layers x 256 experts x (gate, up, down nibbles + E8M0 scales)
    const uint64_t expert_bytes = 47ull * 256 * (2 * (2048ull * 2048 + 2048ull * 128) + 4096ull * 1024 + 4096ull * 64);
    if (m.budget().routed_experts != expert_bytes) std::printf("routed_experts bytes %llu, expected %llu\n", (unsigned long long)m.budget().routed_experts, (unsigned long long)expert_bytes);
    assert(m.budget().routed_experts == expert_bytes);
    std::printf("bind ok: %zu layers, routed experts %.1f GB, attention %.2f GB, dense ffn %.2f GB, embed+head %.2f GB, mtp %.2f GB, vision %.2f GB, audio %.2f GB\n",
                m.layers().size(), m.budget().routed_experts / 1e9, m.budget().attention / 1e9, m.budget().dense_ffn / 1e9,
                (m.budget().embed + m.budget().lm_head) / 1e9, m.budget().mtp / 1e9, m.budget().vision / 1e9, m.budget().audio / 1e9);

    // ---- 3. dequant references ---------------------------------------------------------------
    const char* rd = std::getenv("IE_MIMO26_REF");
    std::string ref = rd && *rd ? rd : "";
    if (ref.empty()) {
        for (const char* cand : {"results/mimo26/p1/ref", "../results/mimo26/p1/ref", "../../results/mimo26/p1/ref"})
            if (fs::exists(std::string(cand) + "/manifest.txt")) { ref = cand; break; }
    }
    if (ref.empty() || !fs::exists(ref + "/manifest.txt")) {
        std::printf("SKIP: config + bind passed, but no reference dump (tools/mimo26/ref_dump.py) -- set IE_MIMO26_REF\n");
        return 77;
    }
    std::ifstream mf(ref + "/manifest.txt");
    std::string line;
    bool ok = true;
    size_t n_entries = 0, n_experts = 0;
    std::vector<float> got;
    while (std::getline(mf, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string kind;
        ss >> kind;
        ++n_entries;
        if (kind == "qkv") {
            uint32_t L, N, K; std::string file;
            ss >> L >> file >> N >> K;
            uint32_t tp = 0;
            const std::string e = mimo26_qkv_dequant_f32(m.layers()[L].qkv, c.q_rows(), c.k_rows(L), c.v_rows(L), c.fp8_block_n, got, &tp);
            if (!e.empty()) { std::printf("  qkv L%u: %s\n", L, e.c_str()); ok = false; continue; }
            if (tp != 4) { std::printf("  qkv L%u: detected tp %u, expected 4\n", L, tp); ok = false; }
            assert(got.size() == size_t(N) * K);
            ok &= same(("qkv L" + std::to_string(L) + " (tp regroup)").c_str(), read_f32(ref + "/" + file), got.data(), got.size());
        } else if (kind == "fp8") {
            std::string role, file; uint32_t L, N, K;
            ss >> role >> L >> file >> N >> K;
            assert(role == "mlp_gate");
            const std::string e = mimo26_fp8_dequant_f32(m.layers()[L].mlp_gate, c.fp8_block_n, c.fp8_block_k, got);
            if (!e.empty()) { std::printf("  fp8 %s L%u: %s\n", role.c_str(), L, e.c_str()); ok = false; continue; }
            assert(got.size() == size_t(N) * K);
            ok &= same(("fp8 " + role + " L" + std::to_string(L)).c_str(), read_f32(ref + "/" + file), got.data(), got.size());
        } else if (kind == "bf16") {
            std::string role, file; uint32_t rows, cols;
            ss >> role >> file >> rows >> cols;
            const Ds41Tensor& t = role == "embed" ? m.embed : role == "lm_head" ? m.lm_head : m.final_norm;
            got.assign(size_t(rows) * cols, 0.f);
            mimo26_bf16_to_f32(t.w->data, rows * cols, got.data());
            ok &= same(("bf16 " + role).c_str(), read_f32(ref + "/" + file), got.data(), got.size());
        } else if (kind == "mxfp4") {
            uint32_t L, E, stride, N, K; std::string proj, full, file;
            ss >> L >> E >> proj >> full >> stride >> file >> N >> K;
            const auto& Lw = m.layers()[L];
            const Ds41Tensor& t = proj == "gate" ? Lw.exp_w1[E] : proj == "up" ? Lw.exp_w3[E] : Lw.exp_w2[E];
            assert(t.w->shape[0] == int64_t(N) && t.w->shape[1] == int64_t(K / 2) && t.s->shape[1] == int64_t(K / 32));
            got.assign(size_t(N) * K, 0.f);
            mimo26_mxfp4_dequant_ref(t.w->data, t.s->data, N, K, got.data());
            ok &= same(("mxfp4 L" + std::to_string(L) + " E" + std::to_string(E) + " " + proj).c_str(), read_f32(ref + "/" + file), got.data(), got.size(), stride);
            ++n_experts;
        } else {
            std::printf("  unknown manifest entry: %s\n", line.c_str());
            ok = false;
        }
    }
    if (n_entries == 0) { std::printf("  the manifest has no entries: nothing was compared\n"); ok = false; }   // P1 gate finding 4
    std::printf("%s: %zu reference entries (%zu expert planes)\n", ok ? "PASS" : "FAIL", n_entries, n_experts);
    return ok ? 0 : 1;
}
