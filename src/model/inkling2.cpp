// src/model/inkling2.cpp — Inkling 2 Mini (`inkling2_loop`) text model, port phase 1: config, safetensors loader and
// a full-sequence forward pass on one card. Correct, not fast: plain kernels from src/ops/inkling2_ops.cpp, the host
// walks the 48 layer calls. Section numbers refer to docs/inkling2/SPEC.md; the numpy twin of this file is
// scripts/inkling2/ref_forward.py (same structure, same rounding switch).
//
// Reused from the first Inkling port (src/ops/inkling_ops.cpp): ink_log_scaling_tau (SPEC 1.5). Phase 1 also used its
// ink_rel_logits; phase 2 forms the relative bias inside i2_attention instead (no [heads, T, context] buffer).
// Everything else differs in formula from P1 (SPEC 13).

#include "ie/inkling2.hpp"

#include "ie/inkling_ops.hpp"

#include "../../third_party/nlohmann/json.hpp"

#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace ie {

using json = nlohmann::json;

// ------------------------------------------------------------------------------------------------- config
std::string inkling2_parse_config(const std::string& json_text, Inkling2Config& c) {
    json root;
    try { root = json::parse(json_text); } catch (const std::exception& e) { return std::string("config.json: ") + e.what(); }
    const json& t = root.contains("text_config") ? root["text_config"] : root;
    std::string err;
    auto u = [&](const char* k, uint32_t& v) {
        if (!t.contains(k) || !t[k].is_number_integer()) { if (err.empty()) err = std::string("config: missing integer ") + k; return; }
        v = t[k].get<uint32_t>();
    };
    auto f = [&](const char* k, float& v) {
        if (!t.contains(k) || !t[k].is_number()) { if (err.empty()) err = std::string("config: missing number ") + k; return; }
        v = t[k].get<float>();
    };
    u("hidden_size", c.hidden); u("num_hidden_layers", c.n_layers); u("vocab_size", c.vocab);
    u("unpadded_vocab_size", c.vocab_unpadded); u("num_attention_heads", c.n_heads); u("num_key_value_heads", c.n_kv_heads);
    u("head_dim", c.head_dim); u("d_rel", c.d_rel); u("rel_extent", c.rel_extent); u("sliding_window_size", c.window);
    u("sconv_kernel_size", c.sconv_k); u("n_layers_per_attn_res_block", c.block); u("n_routed_experts", c.n_experts);
    u("num_experts_per_tok", c.top_k); u("intermediate_size", c.inter);
    f("log_scaling_n_floor", c.log_n_floor); f("log_scaling_alpha", c.log_alpha); f("rms_norm_eps", c.eps);
    f("qk_l2_clip_threshold", c.qk_clip); f("rel_l2_clip_threshold", c.rel_clip); f("route_scale", c.route_scale);
    f("mlp_gate_softcap", c.gate_softcap); f("mlp_up_softcap", c.up_softcap); f("final_logit_softcapping", c.final_softcap);
    if (!err.empty()) return err;
    if (!t.contains("loop_config") || !t["loop_config"].is_object()) return "config: missing loop_config";
    const json& lc = t["loop_config"];
    try {
        c.loop_start = lc.at("core_start").get<uint32_t>(); c.loop_end = lc.at("core_end").get<uint32_t>();
        c.passes = lc.at("num_passes").get<uint32_t>(); c.gate_rank = lc.at("gate_rank").get<uint32_t>();
    } catch (const std::exception& e) { return std::string("config: loop_config: ") + e.what(); }

    // What this runtime implements; anything else is refused rather than run wrongly.
    auto str_or = [&](const char* k, const char* d) { return t.contains(k) && t[k].is_string() ? t[k].get<std::string>() : std::string(d); };
    auto bool_or = [&](const char* k, bool d) { return t.contains(k) && t[k].is_boolean() ? t[k].get<bool>() : d; };
    auto num_or = [&](const char* k, int64_t d) { return t.contains(k) && t[k].is_number_integer() ? t[k].get<int64_t>() : d; };
    if (str_or("scoring_func", "sigmoid") != "sigmoid") return "config: scoring_func must be sigmoid";
    if (num_or("n_shared_experts", 0) != 0) return "config: shared experts not supported";
    if (!bool_or("use_value_norm_no_gamma", true)) return "config: use_value_norm_no_gamma=false not supported";
    if (!bool_or("attn_res_normalize_values", true)) return "config: attn_res_normalize_values=false not supported";
    if (bool_or("q_bias", false) || bool_or("o_bias", false)) return "config: attention biases not supported";
    if (num_or("swa_head_dim", c.head_dim) != c.head_dim || num_or("swa_num_attention_heads", c.n_heads) != c.n_heads ||
        num_or("swa_num_key_value_heads", c.n_kv_heads) != c.n_kv_heads)
        return "config: sliding and global attention shapes differ (not supported)";
    if (c.n_kv_heads == 0 || c.n_heads % c.n_kv_heads) return "config: heads not a multiple of KV heads";
    if (c.head_dim == 0 || c.head_dim > 256) return "config: head_dim must be 1..256";
    if (c.sconv_k != 4) return "config: sconv_kernel_size must be 4";
    if (c.top_k != 8 || c.n_experts == 0 || c.n_experts > 256) return "config: need top-8 of at most 256 experts";
    if (c.block == 0 || c.loop_start % c.block || c.loop_end % c.block || c.loop_start >= c.loop_end ||
        c.loop_end > c.n_layers || c.passes < 1)
        return "config: loop / attn_res block layout not supported";
    if (c.n_layers / c.block + 1 > 7) return "config: too many attn_res blocks for the read-out kernel (max 8 sources)";
    if (c.hidden % 32 || c.inter % 32) return "config: hidden / intermediate must be multiples of 32";
    if (c.vocab_unpadded == 0 || c.vocab_unpadded > c.vocab) return "config: bad unpadded_vocab_size";
    if (t.contains("local_layer_ids") && t["local_layer_ids"].is_array()) {      // cross-check the formula (SPEC 0)
        std::vector<uint32_t> want;
        for (uint32_t L = 0; L < c.n_layers; ++L) if (!c.is_global(L)) want.push_back(L);
        std::vector<uint32_t> got;
        for (const auto& v : t["local_layer_ids"]) got.push_back(v.get<uint32_t>());
        if (got != want) return "config: local_layer_ids disagree with the (L + 1) % block rule";
    }
    return {};
}

std::string inkling2_read_config(const std::string& dir, Inkling2Config& c) {
    std::ifstream f(dir + "/config.json", std::ios::binary);
    if (!f) return "cannot open " + dir + "/config.json";
    std::stringstream ss;
    ss << f.rdbuf();
    return inkling2_parse_config(ss.str(), c);
}

std::string inkling2_read_ablation(const std::string& path, const Inkling2Config& c, Inkling2Ablation& a) {
    a = Inkling2Ablation{};
    a.path = path;
    std::ifstream f(path, std::ios::binary);
    if (!f) return "ablation: cannot open " + path;
    json j;
    try { f >> j; } catch (const std::exception& e) { return "ablation: " + path + ": " + e.what(); }
    const uint32_t H = c.hidden, NC = c.n_calls();
    try {
        if (j.value("format", "") != "ie-ablation-v1") return "ablation: " + path + ": format is not ie-ablation-v1";
        if (j.contains("arch") && j["arch"].get<std::string>() != "inkling2_loop") return "ablation: " + path + ": not for inkling2_loop";
        if (j.at("hidden").get<uint32_t>() != H) return "ablation: hidden size differs from the model's";
        a.rows = j.at("rows").get<uint32_t>();
        std::string blob = j.at("file").get<std::string>();
        if (!blob.empty() && blob[0] != '/') {
            const size_t sl = path.find_last_of('/');
            blob = (sl == std::string::npos ? std::string(".") : path.substr(0, sl)) + "/" + blob;
        }
        std::ifstream b(blob, std::ios::binary | std::ios::ate);
        if (!b) return "ablation: cannot open " + blob;
        if (uint64_t(b.tellg()) != uint64_t(a.rows) * H * 4)
            return "ablation: " + blob + " is not " + std::to_string(a.rows) + " x " + std::to_string(H) + " float32";
        b.seekg(0);
        a.dirs.resize(uint64_t(a.rows) * H);
        b.read(reinterpret_cast<char*>(a.dirs.data()), std::streamsize(a.dirs.size() * 4));
        for (float v : a.dirs) if (!std::isfinite(v)) return "ablation: non-finite direction value";
        std::vector<uint8_t> emb_used(1, 0), attn_used(NC, 0), moe_used(NC, 0);           // remove groups
        std::vector<uint8_t> emb_add(1, 0), attn_add(NC, 0), moe_add(NC, 0);              // add groups
        auto calls = [&](const json& v, std::vector<uint8_t>& dst, std::vector<uint8_t>& used, const char* what) -> std::string {
            dst.assign(NC, 0);
            if (v.is_null()) return {};
            if (v.is_string()) {
                if (v.get<std::string>() != "all") return std::string("ablation: ") + what + ": a list of calls or \"all\"";
                std::fill(dst.begin(), dst.end(), 1);
            } else {
                for (const auto& x : v) {
                    const uint32_t i = x.get<uint32_t>();
                    if (i >= NC) return std::string("ablation: ") + what + ": call " + std::to_string(i) + " out of range";
                    dst[i] = 1;
                }
            }
            for (uint32_t i = 0; i < NC; ++i)
                if (dst[i]) { if (used[i]++) return std::string("ablation: ") + what + " call " + std::to_string(i) + " is in two groups"; }
            return {};
        };
        uint32_t n_attn = 0, n_moe = 0, n_emb = 0, kmax = 0, n_add = 0, a_attn = 0, a_moe = 0, a_emb = 0;
        for (const auto& g : j.at("groups")) {
            Inkling2Ablation::Group G;
            const std::string mode = g.value("mode", std::string("remove"));
            if (mode != "remove" && mode != "add") return "ablation: a group's mode is \"remove\" or \"add\"";
            G.add = mode == "add";
            G.row0 = g.at("rows").at(0).get<uint32_t>();
            G.k = g.at("rows").at(1).get<uint32_t>();
            G.strength = g.value("strength", 1.0f);
            G.embed = g.value("embed", false);
            G.keep_norm = g.value("keep_norm", false);
            if (G.k == 0 || G.k > kI2AblateMaxK || uint64_t(G.row0) + G.k > a.rows) return "ablation: a group's rows are out of range";
            if (!std::isfinite(G.strength)) return "ablation: non-finite strength";
            if (G.embed && (G.add ? emb_add : emb_used)[0]++) return "ablation: the embedding is in two groups of one mode";
            if (auto e = calls(g.contains("attn") ? g["attn"] : json(), G.attn, G.add ? attn_add : attn_used, "attn"); !e.empty()) return e;
            if (auto e = calls(g.contains("moe") ? g["moe"] : json(), G.moe, G.add ? moe_add : moe_used, "moe"); !e.empty()) return e;
            if (G.add) {                                                // steering: any finite vectors
                ++n_add;
                for (uint32_t i = 0; i < NC; ++i) { a_attn += G.attn[i]; a_moe += G.moe[i]; }
                a_emb += G.embed;
                a.groups.push_back(std::move(G));
                continue;
            }
            for (uint32_t x = 0; x < G.k; ++x)                          // orthonormal within the group
                for (uint32_t y = 0; y <= x; ++y) {
                    double d = 0;
                    const float* p = &a.dirs[uint64_t(G.row0 + x) * H];
                    const float* r = &a.dirs[uint64_t(G.row0 + y) * H];
                    for (uint32_t i = 0; i < H; ++i) d += double(p[i]) * r[i];
                    if (std::fabs(d - (x == y ? 1.0 : 0.0)) > 1e-3)
                        return "ablation: group rows " + std::to_string(G.row0) + ".." + std::to_string(G.row0 + G.k - 1) +
                               " are not orthonormal (row " + std::to_string(x) + " . row " + std::to_string(y) + " = " + std::to_string(d) + ")";
                }
            for (uint32_t i = 0; i < NC; ++i) { n_attn += G.attn[i]; n_moe += G.moe[i]; }
            n_emb += G.embed;
            kmax = std::max(kmax, G.k);
            a.groups.push_back(std::move(G));
        }
        if (a.groups.empty()) return "ablation: no groups";
        char s[256];
        std::snprintf(s, sizeof s, "%zu group(s), %u direction(s) (at most %u per write), on the embedding %s, %u of %u attention writes, %u of %u MoE writes",
                      a.groups.size() - n_add, a.rows, kmax, n_emb ? "yes" : "no", n_attn, NC, n_moe, NC);
        a.summary = s;
        if (n_add) {
            std::snprintf(s, sizeof s, "; steering: %u group(s), %u attention writes, %u MoE writes%s", n_add, a_attn, a_moe,
                          a_emb ? ", the embedding" : "");
            a.summary += s;
        }
    } catch (const std::exception& e) { return "ablation: " + path + ": " + e.what(); }
    return {};
}

std::vector<Inkling2Call> inkling2_schedule(const Inkling2Config& c) {
    std::vector<Inkling2Call> s;
    for (uint32_t L = 0; L < c.loop_start; ++L) s.push_back({L, 1});
    for (uint32_t p = 1; p <= c.passes; ++p)
        for (uint32_t L = c.loop_start; L < c.loop_end; ++L) s.push_back({L, p});
    for (uint32_t L = c.loop_end; L < c.n_layers; ++L) s.push_back({L, 1});
    return s;
}

// ------------------------------------------------------------------------------------------------- runtime
namespace {
struct I2Gate { const uint16_t* down = nullptr; const uint16_t* forget = nullptr; const uint16_t* gain = nullptr; };

struct I2Layer {
    // BF16 matrices [out, in]
    const uint16_t *wq = nullptr, *wk = nullptr, *wv = nullptr, *wr = nullptr, *sink = nullptr, *gate_attn = nullptr,
                   *wo = nullptr, *router = nullptr, *kdyn = nullptr, *vdyn = nullptr;
    // fp32 vectors
    const float *nq = nullptr, *nk = nullptr, *nv = nullptr;    // sliding: all three = attn_norm
    const float *mlp_norm = nullptr, *q_scale = nullptr, *rel_scale = nullptr, *rel_proj = nullptr;
    const float *attn_sconv = nullptr, *mlp_sconv = nullptr, *k_conv = nullptr, *v_conv = nullptr;  // [C, 4] or static_w [4, C]
    const float *router_bias = nullptr, *attn_query = nullptr, *mlp_query = nullptr;
    uint32_t     rel_extent = 0;
    I2Gate       ctl[4];                                        // attn (query), k, v, mlp — loop layers only
    I2ExpertBank w13, w2;
};

struct I2Entry {
    float *k = nullptr, *v = nullptr, *ck = nullptr, *cv = nullptr, *ca = nullptr, *cm = nullptr;
    uint32_t ring = 0;                                          // 0: global (row = position); else rows of a ring
};
}  // namespace

// Two cards on one conversation (Inkling2LoadOptions::tp_queue): each rank's thread publishes, per exchange point, the
// event of its push into the other card's buffer and waits (on the host, for the other THREAD's submission only, not
// for the device) for the other rank's event of the same point, then puts a barrier on that event into its own queue.
// Both ranks run the same calls in the same order, so the k-th exchange of one is the k-th of the other.
struct I2TpLink {
    struct Slot { std::atomic<uint64_t> seq{0}; sycl::event ev; };
    Slot ring[2][1024];
    uint64_t n[2] = {0, 0};                                     // exchanges so far, each touched only by its rank
    std::atomic<bool> abort{false};                             // a rank failed: the other leaves at its next exchange
    void* m[2] = {nullptr, nullptr};                            // the ranks' Impl
};

struct Inkling2Model::Impl {
    Inkling2Config       c;
    // ---- tensor parallel (two cards, one conversation); tp_n = 1: one card
    uint32_t             tp_n = 1, tp_r = 0;
    std::shared_ptr<I2TpLink> tp;
    std::unique_ptr<Inkling2Model> tp_peer;                     // rank 0 only: the other card's model
    std::thread          tp_thr;                                // rank 0 only: runs the peer's calls
    std::mutex           tp_mu;
    std::condition_variable tp_cv;
    std::function<void()> tp_job;
    std::atomic<int>     tp_state{0};                           // 0 idle, 1 job posted, 2 job done
    bool                 tp_quit = false;
    bool                 tp_in = false;                         // rank 0 inside its own half of a forward
    Impl* peer() const { return static_cast<Impl*>(tp->m[1 - tp_r]); }
    uint32_t* tp_flag = nullptr;                                // [2]: the other card's last k, a give-up marker
    uint32_t nh_l() const { return c.n_heads / tp_n; }
    uint32_t nkv_l() const { return c.n_kv_heads / tp_n; }
    uint64_t kd_l() const { return uint64_t(nkv_l()) * c.head_dim; }
    // The exchange after a push into the other card (or, with no data, a plain meeting point). Default: device flags --
    // this card writes the exchange number k into the other card's flag word behind its data (same in-order queue), and
    // waits in a one-item kernel until the other card's k has landed in its own flag word; the host never waits.
    // (A queue barrier on the other card's event does NOT order the reads on this driver: measured wrong values.)
    // IE_INKLING2_TP_SYNC=host: the host waits for the other card's push event instead (correct, ~2 ms a token slower).
    void tp_sync(sycl::event mine) {
        I2TpLink& L = *tp;
        const uint64_t k = ++L.n[tp_r];
        static const bool host = [] { const char* e = std::getenv("IE_INKLING2_TP_SYNC"); return e && std::string(e) == "host"; }();
        static const bool nosync = std::getenv("IE_INKLING2_TP_NOSYNC") != nullptr;   // TIMING ONLY: wrong values
        if (nosync) return;
        if (!host) {
            q->fill<uint32_t>(peer()->tp_flag, uint32_t(k), 1);
            uint32_t* f = tp_flag;
            uint32_t* bad = tp_flag + 1;
            const uint32_t kk = uint32_t(k);
            q->single_task([=] {
                sycl::atomic_ref<uint32_t, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                                 sycl::access::address_space::global_space> a(*f);
                for (uint32_t i = 0; i < (1u << 24); ++i) if (a.load() >= kk) return;
                *bad = kk;                                              // gave up (the other card never came): no hang
            });
            return;
        }
        I2TpLink::Slot& my = L.ring[tp_r][k & 1023];
        my.ev = mine;
        my.seq.store(k, std::memory_order_release);
        I2TpLink::Slot& pe = L.ring[1 - tp_r][k & 1023];
        for (uint32_t i = 0; pe.seq.load(std::memory_order_acquire) != k; ++i) {
            if (L.abort.load(std::memory_order_relaxed)) throw std::runtime_error("the other card's forward failed");
            if (i > 256) std::this_thread::yield();
        }
        pe.ev.wait();
        q->ext_oneapi_submit_barrier({pe.ev});
    }
    // after a forward: 0, or the exchange a device wait gave up on (both ranks then start over)
    uint32_t tp_gave_up() { uint32_t v = 0; q->memcpy(&v, tp_flag + 1, 4).wait(); return v; }
    void tp_reset_flags() { q->memset(tp_flag, 0, 8).wait(); }
    void tp_push(void* peer_dst, const void* src, size_t bytes) { tp_sync(q->memcpy(peer_dst, src, bytes)); }
    void tp_post(std::function<void()> f) {
        { std::lock_guard<std::mutex> lk(tp_mu); tp_job = std::move(f); tp_state.store(1, std::memory_order_release); }
        tp_cv.notify_one();
    }
    void tp_wait() {
        for (uint32_t i = 0; tp_state.load(std::memory_order_acquire) != 2; ++i) if (i > 256) std::this_thread::yield();
        tp_state.store(0, std::memory_order_relaxed);
    }
    void tp_worker() {
        for (;;) {
            std::function<void()> f;
            {
                std::unique_lock<std::mutex> lk(tp_mu);
                tp_cv.wait(lk, [&] { return tp_quit || tp_state.load(std::memory_order_acquire) == 1; });
                if (tp_quit) return;
                f = std::move(tp_job);
            }
            f();
            tp_state.store(2, std::memory_order_release);
        }
    }
    sycl::queue*         q = nullptr;
    bool                 R = false;                             // reference bf16 emulation
    SafetensorsModel     st;
    const SafeTensorInfo* embed = nullptr;                      // host-side (mmap)
    std::vector<I2Layer> layers;
    const float*         embed_norm = nullptr;
    const float*         final_norm = nullptr;
    const float*         output_query = nullptr;
    std::vector<const float*> msg_queries;                      // per loop layer: [4, H] rows m_q, m_k, m_v, m_mlp
    const uint16_t*      unembed = nullptr;                     // [vocab_unpadded, H]
    float*               all_buf = nullptr;                     // Inkling2RunOptions::all_logits
    uint64_t             all_cap = 0;

    std::vector<void*>   weight_allocs;
    size_t               weight_bytes = 0;

    // per-(layer, pass) state, SPEC 6
    std::vector<I2Entry> entries;
    std::vector<void*>   fixed_allocs, glob_allocs;         // rings + conv histories (allocated once) / global K, V
    size_t               fixed_bytes = 0, glob_bytes = 0;
    uint32_t             cap = 0, npos = 0;                     // global rows allocated, positions cached
    uint32_t             max_chunk = 256, rollback = 256, crow = 259;   // crow = conv input ring rows (rollback + 3)
    bool                 kv16 = false;                          // attention keys / values cached in bf16 (the state in use)
    bool                 kv16_single = false, kv16_batch = false;   // the one-conversation state's, the batch slots'
    std::string          kv_why;
    uint64_t kvb() const { return kv16 ? 2 : 4; }               // bytes per cached key / value element
    float* kvdev(std::vector<void*>& list, size_t& bytes, uint64_t n) {   // a key / value array (bf16 or fp32 elements)
        if (kv16) return reinterpret_cast<float*>(dev<uint16_t>(list, bytes, n));
        return dev<float>(list, bytes, n);
    }
    uint32_t             hw = 0;                                // furthest position ever written + 1
    static constexpr uint64_t kPartFloats = 16ull << 20;        // split-attention partials (64 MiB)
    bool                 scratch_teacher = false;

    std::vector<void*>   scratch_allocs;
    size_t               scratch_bytes = 0;

    // the ablation set (Inkling2Ablation): per write, the group's directions on the device (k = 0: none)
    struct AblW { const float* d = nullptr; uint32_t k = 0; float s = 0.f; bool keep_norm = false; };
    std::vector<void*>   abl_allocs;
    size_t               abl_bytes = 0;
    std::vector<AblW>    abl_attn, abl_moe;                     // [n_calls]
    AblW                 abl_emb;
    std::vector<const float*> add_attn, add_moe;                // [n_calls]: steering vectors (strength * sum_k r_k) or null
    const float*         add_emb = nullptr;
    std::string          abl_summary;
    // the recorder (Inkling2RunOptions::record_last): forward rows >= rec_first go to rec
    std::vector<float>*  rec = nullptr;
    uint32_t             rec_first = 0;

    // ---- batched decoding (Inkling2Model::batch_*): per-slot state in one allocation per entry array, slot b's arrays
    // b * stride floats after slot 0's (I2Rows); a slot's prefill swaps its entries in for the single state's
    struct Slot { std::vector<I2Entry> e; uint32_t npos = 0, hw = 0; };
    std::vector<Slot>    bslots;
    std::vector<I2Entry> bbase;                                 // slot 0's pointers per entry
    std::vector<void*>   batch_allocs;
    size_t               batch_bytes = 0;
    uint32_t             bcap = 0;                              // global rows per slot
    uint32_t             bchunk = 64, broll = 64, bcrow = 67;   // the slots' prefill chunk, rewind reach, conv ring rows
    float               *bpart = nullptr, *bfn = nullptr, *bvocab = nullptr;
    // sampling on the card (batch_step_sample): device copies of the rows' parameters / ids, the candidate scratch
    struct BSamp { I2SampleRow* rows = nullptr; int32_t *rec = nullptr, *pfi = nullptr, *cand = nullptr, *out = nullptr; float* pfd = nullptr;
                   uint32_t nrows = 0, nrec = 0, npf = 0; };
    BSamp bs;
    bool bsamp_on = false;
    std::vector<int32_t> bsamp_ids;
    void bs_free() {
        if (!q) return;
        for (void* p : {(void*)bs.rows, (void*)bs.rec, (void*)bs.pfi, (void*)bs.cand, (void*)bs.out, (void*)bs.pfd}) if (p) sycl::free(p, *q);
        bs = BSamp{};
    }
    int32_t             *brpos = nullptr, *brslot = nullptr;
    bool                 bm = false;                            // a batched step is running
    uint32_t             bmaxpos = 0;                           // the largest position among its rows
    std::vector<int32_t> bpos_host;                             // its rows' positions

    ~Impl() {
        if (tp_thr.joinable()) { { std::lock_guard<std::mutex> lk(tp_mu); tp_quit = true; } tp_cv.notify_one(); tp_thr.join(); }
        tp_peer.reset();
        bs_free();
        free_list(batch_allocs); free_list(abl_allocs); free_list(scratch_allocs); free_list(glob_allocs); free_list(fixed_allocs); free_list(weight_allocs);
        if (all_buf && q) sycl::free(all_buf, *q);
    }

    void free_list(std::vector<void*>& v) {
        if (q) for (void* p : v) sycl::free(p, *q);
        v.clear();
    }
    template <class T> T* dev(std::vector<void*>& list, size_t& bytes, size_t n) {
        T* p = sycl::malloc_device<T>(n ? n : 1, *q);
        if (!p) throw std::runtime_error("device allocation of " + std::to_string(n * sizeof(T)) + " bytes failed");
        list.push_back(p);
        bytes += n * sizeof(T);
        return p;
    }

    // ---- loading
    const SafeTensorInfo* need(const std::string& name, const char* dtype, std::vector<int64_t> shape) {
        const SafeTensorInfo* t = st.find("model.llm." + name);
        if (!t) throw std::runtime_error("missing tensor " + name);
        if (t->dtype_str != dtype) throw std::runtime_error(name + ": dtype " + t->dtype_str + ", expected " + dtype);
        if (t->shape != shape) {
            std::string s;
            for (auto d : t->shape) s += std::to_string(d) + " ";
            throw std::runtime_error(name + ": unexpected shape [ " + s + "]");
        }
        return t;
    }
    bool has(const std::string& name) const { return st.find("model.llm." + name) != nullptr; }

    const uint16_t* up_bf16(const std::string& name, std::vector<int64_t> shape, uint64_t rows_keep = 0) {
        const SafeTensorInfo* t = need(name, "BF16", shape);
        uint64_t n = t->numel();
        if (rows_keep) n = rows_keep * (n / uint64_t(shape[0]));
        auto* d = dev<uint16_t>(weight_allocs, weight_bytes, n);
        q->memcpy(d, t->data, n * 2).wait();
        return d;
    }
    std::vector<float> host_f32(const std::string& name, const std::vector<int64_t>& shape) {
        const SafeTensorInfo* t = st.find("model.llm." + name);
        if (!t) throw std::runtime_error("missing tensor " + name);
        need(name, t->dtype_str == "F32" ? "F32" : "BF16", shape);
        std::vector<float> v(t->numel());
        if (t->dtype_str == "F32") std::memcpy(v.data(), t->data, v.size() * 4);
        else for (size_t i = 0; i < v.size(); ++i) {
            uint16_t b; std::memcpy(&b, t->data + 2 * i, 2);
            v[i] = i2_bf16_to_f32(b);
        }
        return v;
    }
    const float* up_vec(const std::vector<float>& v) {
        auto* d = dev<float>(weight_allocs, weight_bytes, v.size());
        q->memcpy(d, v.data(), v.size() * 4).wait();
        return d;
    }
    const float* up_f32(const std::string& name, std::vector<int64_t> shape) { return up_vec(host_f32(name, shape)); }

    I2ExpertBank up_bank(uint32_t L, const char* which, uint32_t rows, uint32_t cols) {
        const std::string p = "layers." + std::to_string(L) + ".mlp.experts." + which + "_weight";
        const int64_t E = c.n_experts;
        I2ExpertBank b;
        b.rows = rows; b.cols = cols;
        auto raw = [&](const SafeTensorInfo* t) {
            auto* d = dev<uint8_t>(weight_allocs, weight_bytes, t->nbytes);
            q->memcpy(d, t->data, t->nbytes).wait();
            return d;
        };
        if (has(p + "_amax")) {                                                 // NVFP4, SPEC 10.1
            b.fmt = 0;
            b.w = raw(need(p, "U8", {E, rows, cols / 2}));
            b.s = raw(need(p + "_scale", "F8_E4M3", {E, rows, cols / 16}));
            std::vector<float> amax = host_f32(p + "_amax", {E});
            const float den = host_f32(p + "_outer_scale_denominator", {1})[0];
            std::vector<float> fac(E);
            for (int64_t e = 0; e < E; ++e)                                     // the reference rounds both to bf16 first
                fac[e] = R ? i2_bf16_round(amax[e]) / i2_bf16_round(den) : amax[e] / den;
            b.fac = up_vec(fac);
        } else {                                                                // MXFP8, SPEC 10.2
            b.fmt = 1;
            b.w = raw(need(p, "F8_E4M3", {E, rows, cols}));
            b.s = raw(need(p + "_scale", "U8", {E, rows, cols / 32}));
        }
        return b;
    }

    void load_all(const std::string& dir) {
        const int64_t H = c.hidden, NH = c.n_heads, NKV = c.n_kv_heads, HD = c.head_dim, DR = c.d_rel;
        const int64_t QD = NH * HD, KD = NKV * HD;
        st.advise(MADV_SEQUENTIAL);
        embed = need("embed.weight", "BF16", {int64_t(c.vocab), H});
        embed_norm = up_f32("embed_norm.weight", {H});
        final_norm = up_f32("norm.weight", {H});
        output_query = up_f32("attn_res.output_query", {H});
        unembed = up_bf16("unembed.weight", {int64_t(c.vocab), H}, c.vocab_unpadded);
        {   // SPEC 5.3: loop layer i: m_q = queries[2i], m_mlp = queries[2i+1], m_k = kv[2i], m_v = kv[2i+1]
            const int64_t n2 = 2 * int64_t(c.n_loop());
            auto a = host_f32("middle_cycle_cross_pass_queries", {n2, H});
            auto b = host_f32("middle_cycle_cross_pass_kv_queries", {n2, H});
            for (uint32_t i = 0; i < c.n_loop(); ++i) {
                std::vector<float> qv(4 * H);
                std::memcpy(&qv[0 * H], &a[(2 * i) * H], H * 4);
                std::memcpy(&qv[1 * H], &b[(2 * i) * H], H * 4);
                std::memcpy(&qv[2 * H], &b[(2 * i + 1) * H], H * 4);
                std::memcpy(&qv[3 * H], &a[(2 * i + 1) * H], H * 4);
                msg_queries.push_back(up_vec(qv));
            }
        }
        layers.resize(c.n_layers);
        for (uint32_t L = 0; L < c.n_layers; ++L) {
            I2Layer& l = layers[L];
            const std::string p = "layers." + std::to_string(L) + ".";
            const bool glob = c.is_global(L);
            l.wq = up_bf16(p + "attn.wq_du.weight", {QD, H});
            l.wk = up_bf16(p + "attn.wk_dv.weight", {KD, H});
            l.wv = up_bf16(p + "attn.wv_dv.weight", {KD, H});
            l.wr = up_bf16(p + "attn.wr_du.weight", {NH * DR, H});
            l.sink = up_bf16(p + "attn.attn_sink_proj.weight", {NH, H});
            l.gate_attn = up_bf16(p + "attn.gate_attn.weight", {QD, H});
            l.wo = up_bf16(p + "attn.wo_ud.weight", {H, QD});
            l.q_scale = up_f32(p + "attn.q_l2_clip.scale", {NH, HD});
            l.rel_scale = up_f32(p + "attn.rel_l2_clip.scale", {NH, DR});
            const SafeTensorInfo* rp = st.find("model.llm." + p + "attn.rel_logits_proj.proj");
            if (!rp || rp->shape.size() != 2) throw std::runtime_error(p + "attn.rel_logits_proj.proj missing");
            l.rel_extent = uint32_t(rp->shape[1]);
            l.rel_proj = up_f32(p + "attn.rel_logits_proj.proj", {DR, int64_t(l.rel_extent)});
            if (glob) {
                l.nq = up_f32(p + "q_input_norm.weight", {H});
                l.nk = up_f32(p + "k_input_norm.weight", {H});
                l.nv = up_f32(p + "v_input_norm.weight", {H});
                l.kdyn = up_bf16(p + "attn.k_sconv.weight_proj", {NKV * 4, H});
                l.vdyn = up_bf16(p + "attn.v_sconv.weight_proj", {NKV * 4, H});
                l.k_conv = up_f32(p + "attn.k_sconv.static_w", {4, KD});
                l.v_conv = up_f32(p + "attn.v_sconv.static_w", {4, KD});
            } else {
                l.nq = l.nk = l.nv = up_f32(p + "attn_norm.weight", {H});
                l.k_conv = up_f32(p + "attn.k_sconv.weight", {KD, 1, 4});
                l.v_conv = up_f32(p + "attn.v_sconv.weight", {KD, 1, 4});
            }
            l.attn_sconv = up_f32(p + "attn_sconv.weight", {H, 1, 4});
            l.mlp_sconv = up_f32(p + "mlp_sconv.weight", {H, 1, 4});
            l.mlp_norm = up_f32(p + "mlp_norm.weight", {H});
            l.router = up_bf16(p + "mlp.gate.weight", {int64_t(c.n_experts), H});
            l.router_bias = up_f32(p + "mlp.gate.bias", {int64_t(c.n_experts)});
            if (L > 0)                                                          // trap 1: attn_queries.{i} is layer i + 1
                l.attn_query = up_f32("attn_res.attn_queries." + std::to_string(L - 1),
                                      glob ? std::vector<int64_t>{3, H} : std::vector<int64_t>{H});
            l.mlp_query = up_f32("attn_res.mlp_queries." + std::to_string(L), {H});
            if (c.is_loop(L)) {
                const char* names[4] = {"attn", "k", "v", "mlp"};
                const int64_t G = c.gate_rank;
                for (int g = 0; g < 4; ++g) {
                    const std::string cp = p + "controllers." + names[g] + ".";
                    l.ctl[g].down = up_bf16(cp + "down_weight", {G, 2 * H});
                    l.ctl[g].forget = up_bf16(cp + "forget_weight", {H, G});
                    l.ctl[g].gain = up_bf16(cp + "gain_weight", {H, G});
                }
            }
            l.w13 = up_bank(L, "w13", 2 * c.inter, c.hidden);
            l.w2 = up_bank(L, "w2", c.hidden, c.inter);
            if (tp_n > 1) {                     // this rank's heads: row blocks of the full copies (head-major rows)
                const int64_t r = tp_r, nh = NH / tp_n, nkv = NKV / tp_n, KDh = nkv * HD;
                l.wq += r * nh * HD * H; l.gate_attn += r * nh * HD * H; l.wr += r * nh * DR * H; l.sink += r * nh * H;
                l.wk += r * KDh * H; l.wv += r * KDh * H;
                l.q_scale += r * nh * HD; l.rel_scale += r * nh * DR;
                if (glob) {
                    l.kdyn += r * nkv * 4 * H; l.vdyn += r * nkv * 4 * H;
                    for (int kv = 0; kv < 2; ++kv) {          // static_w [4, KD]: this rank's columns, repacked
                        auto w = host_f32(p + (kv ? "attn.v_sconv.static_w" : "attn.k_sconv.static_w"), {4, KD});
                        std::vector<float> h(4 * KDh);
                        for (int64_t j = 0; j < 4; ++j) std::memcpy(&h[j * KDh], &w[j * KD + r * KDh], KDh * 4);
                        (kv ? l.v_conv : l.k_conv) = up_vec(h);
                    }
                } else {
                    l.k_conv += r * KDh * 4; l.v_conv += r * KDh * 4;      // [KD, 1, 4]
                }
            }
        }
        (void)dir;
    }

    // ---- state (SPEC 6)
    uint32_t entry_layer(uint32_t i) const { return i < c.n_layers ? i : c.loop_start + (i - c.n_layers) % c.n_loop(); }
    void init_state() {                                                     // rings and conv histories, once
        const uint64_t KD = kd_l(), H = c.hidden;
        entries.assign(c.n_calls(), I2Entry{});
        for (uint32_t i = 0; i < c.n_calls(); ++i) {
            I2Entry& e = entries[i];
            if (!c.is_global(entry_layer(i))) {
                e.ring = c.window + rollback;
                e.k = kvdev(fixed_allocs, fixed_bytes, uint64_t(e.ring) * KD);
                e.v = kvdev(fixed_allocs, fixed_bytes, uint64_t(e.ring) * KD);
            }
            e.ck = dev<float>(fixed_allocs, fixed_bytes, crow * KD); e.cv = dev<float>(fixed_allocs, fixed_bytes, crow * KD);
            e.ca = dev<float>(fixed_allocs, fixed_bytes, crow * H);  e.cm = dev<float>(fixed_allocs, fixed_bytes, crow * H);
        }
    }
    void grow_global(uint32_t new_cap) {                                    // keeps the cached rows
        const uint64_t KD = kd_l();
        std::vector<void*> nl;
        size_t nb = 0;
        std::vector<std::pair<float*, float*>> nkv(entries.size(), {nullptr, nullptr});
        for (size_t i = 0; i < entries.size(); ++i) {
            if (entries[i].ring) continue;
            nkv[i].first = kvdev(nl, nb, uint64_t(new_cap) * KD);
            nkv[i].second = kvdev(nl, nb, uint64_t(new_cap) * KD);
            if (npos) {
                q->memcpy(nkv[i].first, entries[i].k, uint64_t(npos) * KD * kvb());
                q->memcpy(nkv[i].second, entries[i].v, uint64_t(npos) * KD * kvb());
            }
        }
        q->wait();
        free_list(glob_allocs);
        for (size_t i = 0; i < entries.size(); ++i)
            if (!entries[i].ring) { entries[i].k = nkv[i].first; entries[i].v = nkv[i].second; }
        glob_allocs = std::move(nl);
        glob_bytes = nb;
        cap = new_cap;
    }

    // ---- per-chunk scratch, sized for max_chunk once (dry: count the bytes only)
    struct S {
        float *x0raw, *x0, *A, *G[3], *qin, *kin, *vin, *qb, *kp, *vp, *kc, *vc, *dyn, *dyn2, *rb, *sk, *gt, *o, *attn,
              *x, *res, *U, *U2, *mi, *logits, *w, *margin, *gu, *act, *ey, *moe, *y, *tau, *cat, *gd, *gf, *gg,
              *M, *F, *fn, *vocab, *part;
        int32_t *qpos, *used, *own, *forced, *order, *tiles, *ntiles, *acnt;
        float *apart;
        float *ofull, *opeer;                                           // tensor parallel: both ranks' heads [T, QD]; the other's
        std::vector<float*> out, teach;
    } s{};

    size_t alloc_scratch(uint32_t T, bool teacher, bool dry) {
        if (!dry) { free_list(scratch_allocs); scratch_bytes = 0; }
        size_t counted = 0;
        const uint64_t H = c.hidden, QD = uint64_t(c.n_heads) * c.head_dim, KD = uint64_t(c.n_kv_heads) * c.head_dim;
        const uint64_t K8 = uint64_t(T) * c.top_k;
        auto F = [&](uint64_t n) -> float* { if (dry) { counted += n * 4; return nullptr; } return dev<float>(scratch_allocs, scratch_bytes, n); };
        auto I = [&](uint64_t n) -> int32_t* { if (dry) { counted += n * 4; return nullptr; } return dev<int32_t>(scratch_allocs, scratch_bytes, n); };
        s.x0raw = F(T * H); s.x0 = F(T * H); s.A = F(3 * T * H);
        for (auto& g : s.G) g = F(T * H);
        s.qin = F(T * H); s.kin = F(T * H); s.vin = F(T * H);
        s.qb = F(T * QD); s.kp = F(T * KD); s.vp = F(T * KD); s.kc = F(T * KD); s.vc = F(T * KD);
        s.dyn = F(T * uint64_t(c.n_kv_heads) * 4); s.dyn2 = F(T * uint64_t(c.n_kv_heads) * 4); s.rb = F(T * uint64_t(c.n_heads) * c.d_rel);
        s.sk = F(T * uint64_t(c.n_heads));
        s.gt = F(T * QD); s.o = F(T * QD); s.attn = F(T * H); s.x = F(T * H); s.res = F(T * H);
        s.U = F(T * H); s.U2 = F(T * H); s.mi = F(T * H);
        s.logits = F(T * uint64_t(c.n_experts)); s.w = F(K8); s.margin = F(T);
        s.gu = F(K8 * 2 * c.inter); s.act = F(K8 * c.inter); s.ey = F(K8 * H); s.moe = F(T * H); s.y = F(T * H);
        s.tau = F(T);
        s.cat = F(3 * T * 2 * H); s.gd = F(3 * T * uint64_t(c.gate_rank)); s.gf = F(3 * T * H); s.gg = F(3 * T * H);   // 3 gates at once
        s.apart = F(uint64_t(T) * ((H + 255) / 256) * 5);
        s.M = F(4 * T * H); s.F = F(T * H); s.fn = F(H); s.vocab = F(c.vocab_unpadded);
        s.part = F(kPartFloats);
        s.ofull = tp_n > 1 ? F(T * QD) : nullptr;
        s.opeer = tp_n > 1 ? F(T * QD / 2) : nullptr;
        if (tp_n > 1 && !dry && !tp_flag) { tp_flag = dev<uint32_t>(fixed_allocs, fixed_bytes, 2); q->memset(tp_flag, 0, 8).wait(); }
        s.qpos = I(T); s.used = I(K8); s.own = I(K8); s.forced = I(K8);
        s.order = I(K8); s.tiles = I(3 * (K8 / kI2MoeTile + c.n_experts + 1)); s.ntiles = I(1);
        s.acnt = I(T);
        if (!dry) q->memset(s.acnt, 0, uint64_t(T) * 4).wait();          // i2_write_ablate's tickets start at zero
        s.out.assign(c.n_calls(), nullptr);
        s.teach.assign(c.n_calls(), nullptr);
        for (auto& p : s.out) p = F(T * H);
        if (teacher) for (auto& p : s.teach) p = F(T * H);
        if (!dry) scratch_teacher = teacher;
        return dry ? counted : scratch_bytes;
    }

    // ---- ops in model terms
    // n controller gates at once (fp32 path): the norms, down-projections, forget / gain pairs and combines are one
    // launch each for all n; the same values as n gate() calls.
    void gates(uint32_t n, const I2Gate* const* g, const float* const* h, const float* const* m, uint32_t T, float* const* out) {
        const uint32_t H = c.hidden, G = c.gate_rank;
        const uint64_t TH = uint64_t(T) * H;
        I2NormSeg ns[6];
        I2LinSeg dn[3], fg[6];
        I2GateSeg cb[3];
        for (uint32_t i = 0; i < n; ++i) {
            float* cat = s.cat + i * 2 * TH;
            float* gd = s.gd + uint64_t(i) * T * G;
            float* gf = s.gf + i * TH;
            float* gg = s.gg + i * TH;
            ns[2 * i] = {h[i], nullptr, cat};
            ns[2 * i + 1] = {m[i], nullptr, cat + H};
            dn[i] = {cat, g[i]->down, G, gd};
            fg[2 * i] = {gd, g[i]->forget, H, gf};
            fg[2 * i + 1] = {gd, g[i]->gain, H, gg};
            cb[i] = {h[i], m[i], gf, gg, out[i]};
        }
        static const bool gfused = [] { const char* e = std::getenv("IE_INKLING2_GATE_FUSED"); return e && *e == '1'; }();
        if (gfused && n <= 3 && G % 8 == 0 && G <= 128 && H % 8 == 0) {   // two launches instead of four
            I2GateDownSeg ds[3];
            I2GateUpSeg us[3];
            for (uint32_t i = 0; i < n; ++i) {
                ds[i] = {h[i], m[i], g[i]->down, dn[i].y};
                us[i] = {dn[i].y, g[i]->forget, g[i]->gain, h[i], m[i], out[i]};
            }
            i2_gate_down_norm(*q, n, ds, T, H, G, c.eps);
            i2_gate_up_combine(*q, n, us, T, H, G);
            return;
        }
        i2_rmsnorm_multi(*q, 2 * n, ns, T, H, c.eps, 2 * H, false);
        i2_linear_bf16_multi(*q, n, dn, T, 2 * H, false, true, bm);
        i2_linear_bf16_multi(*q, 2 * n, fg, T, G, false, true, bm);
        i2_gate_combine_multi(*q, n, cb, TH);
    }

    void gate(const I2Gate& g, const float* h, const float* m, uint32_t T, float* out) {    // SPEC 1.11
        const uint32_t H = c.hidden;
        if (!R) { const I2Gate* gp = &g; gates(1, &gp, &h, &m, T, &out); return; }
        const I2NormSeg ns2[2] = {{h, nullptr, s.cat}, {m, nullptr, s.cat + H}};
        i2_rmsnorm_multi(*q, 2, ns2, T, H, c.eps, 2 * H, R);
        i2_linear_bf16(*q, s.cat, T, 2 * H, g.down, c.gate_rank, s.gd, R, !R);
        const I2LinSeg fg[2] = {{s.gd, g.forget, H, s.gf}, {s.gd, g.gain, H, s.gg}};
        i2_linear_bf16_multi(*q, 2, fg, T, c.gate_rank, false, !R);
        i2_gate_combine(*q, h, m, s.gf, s.gg, uint64_t(T) * H, out);
    }

    static uint32_t attn_min_keys() {                                   // keys per split (IE_INKLING2_ATTN_MINKEYS for tuning)
        static const uint32_t v = [] { const char* e = std::getenv("IE_INKLING2_ATTN_MINKEYS"); return e ? uint32_t(std::atoi(e)) : 128u; }();
        return v ? v : 128u;
    }
    static bool three_pass() { static const bool v = std::getenv("IE_INKLING2_ATTN_3PASS") != nullptr; return v; }
    void attention(uint32_t L, const I2Entry& E, uint32_t T, uint32_t pos0, float* out) {   // SPEC 3
        const I2Layer& l = layers[L];
        const bool glob = c.is_global(L);
        const uint32_t H = c.hidden, NH = nh_l(), NKV = nkv_l(), HD = c.head_dim, QD = NH * HD, KD = NKV * HD;
        // every projection of the three inputs in one launch (fast path): q, k, v, relative states (5), sink (6),
        // output gate (9), and the dynamic conv weights of a global layer; the same values as one call each
        I2LinSeg seg[8] = {{s.qin, l.wq, QD, s.qb},          {s.kin, l.wk, KD, s.kp},          {s.vin, l.wv, KD, s.vp},
                           {s.qin, l.wr, NH * c.d_rel, s.rb}, {s.qin, l.sink, NH, s.sk},        {s.qin, l.gate_attn, QD, s.gt}};
        uint32_t ns = 6;
        if (glob) { seg[ns++] = {s.kin, l.kdyn, NKV * 4, s.dyn}; seg[ns++] = {s.vin, l.vdyn, NKV * 4, s.dyn2}; }
        i2_linear_bf16_multi(*q, ns, seg, T, H, R, !R, bm);
        static const bool unfused = std::getenv("IE_INKLING2_UNFUSED") != nullptr;            // A/B switch
        const bool split = !(R || three_pass() || HD > 128 || NH / NKV > 4);
        if (!R && !unfused && HD <= 128 && split) {
            I2AttnPrep p;
            p.T = T; p.pos0 = pos0; p.nh = NH; p.nkv = NKV; p.hd = HD; p.d_rel = c.d_rel; p.crow = crow; p.ring = E.ring;
            p.glob = glob; p.qk_clip = c.qk_clip; p.rel_clip = c.rel_clip; p.eps = c.eps;
            p.qb = s.qb; p.rb = s.rb; p.q_scale = l.q_scale; p.rel_scale = l.rel_scale; p.tau = s.tau;
            p.kp = s.kp; p.vp = s.vp; p.dyn_k = s.dyn; p.dyn_v = s.dyn2; p.k_conv = l.k_conv; p.v_conv = l.v_conv;
            p.ck = E.ck; p.cv = E.cv; p.Kc = E.k; p.Vc = E.v; p.kv16 = kv16;
            I2Rows rows;
            uint64_t kvs = 0;
            uint32_t nsm = 0;
            if (bm) {                                                   // batched: each row its own slot and position
                rows = {brpos, brslot};
                kvs = uint64_t(E.ring ? E.ring : bcap) * KD;
                nsm = i2_attention_splits(1, bmaxpos, NH, HD, glob ? 0 : c.window, kPartFloats, attn_min_keys());
                p.rows = rows; p.kv_stride = kvs; p.c_stride = uint64_t(crow) * KD;
            }
            i2_attn_prep(*q, p);
            // tensor parallel, one token: this rank's heads straight into its block of the full row
            float* o = tp_n > 1 && T == 1 ? s.ofull + uint64_t(tp_r) * QD : s.o;
            i2_attention_split(*q, s.qb, E.k, E.v, E.ring, s.rb, l.rel_proj, c.d_rel, l.rel_extent, s.sk, o, T, pos0, NH,
                               NKV, HD, glob ? 0 : c.window, bm ? bpart : s.part, kPartFloats, attn_min_keys(), s.gt,
                               rows, kvs, nsm, kv16);                   // + step 9 gate
            if (tp_n > 1) {                                             // join the heads; then the output projection
                const uint32_t QDF = c.n_heads * HD, Hh = H / tp_n;
                Impl* P = peer();
                if (T == 1) tp_push(P->s.ofull + uint64_t(tp_r) * QD, o, uint64_t(QD) * 4);
                else {                                                  // a chunk: one contiguous push, then interleave
                    tp_push(P->s.opeer, s.o, uint64_t(T) * QD * 4);
                    float* of = s.ofull;
                    const float *mine = s.o, *other = s.opeer;
                    const uint32_t r = tp_r;
                    q->parallel_for(sycl::range<1>(uint64_t(T) * QDF), [=](sycl::id<1> id) {
                        const uint64_t t = id[0] / QDF, j = id[0] % QDF, blk = j / QD, i = j % QD;
                        of[id[0]] = (blk == r ? mine : other)[t * QD + i];
                    });
                }
                if (T == 1 && out == s.attn) {                          // one token: each rank its half of the rows
                    i2_linear_bf16(*q, s.ofull, 1, QDF, l.wo + uint64_t(tp_r) * Hh * QDF, Hh, out + uint64_t(tp_r) * Hh, false, true, bm);
                    tp_push(P->s.attn + uint64_t(tp_r) * Hh, out + uint64_t(tp_r) * Hh, uint64_t(Hh) * 4);
                } else i2_linear_bf16(*q, s.ofull, T, QDF, l.wo, H, out, false, true, bm);
                return;
            }
            i2_linear_bf16(*q, s.o, T, QD, l.wo, H, out, false, true, bm);
            return;
        }
        // 1. query: scale -> L2 clip -> (global) tau
        i2_scale_l2clip(*q, s.qb, T * NH, HD, l.q_scale, NH, c.qk_clip, s.qb, R);
        if (glob) i2_mul_token(*q, s.qb, T, QD, s.tau, s.qb, R);
        // 2./3. key, value: conv -> clip / unweighted norm, written into this entry's cache at pos0
        if (glob) i2_sconv_dynamic(*q, s.kp, T, KD, s.dyn, l.k_conv, NKV, E.ck, crow, pos0, s.kc, R);
        else i2_sconv_static(*q, s.kp, T, KD, l.k_conv, E.ck, crow, pos0, s.kc, R);
        i2_scale_l2clip(*q, s.kc, T * NKV, HD, nullptr, NKV, c.qk_clip, s.kc, R);
        i2_store_rows(*q, s.kc, T, KD, E.k, pos0, E.ring);
        if (glob) i2_sconv_dynamic(*q, s.vp, T, KD, s.dyn2, l.v_conv, NKV, E.cv, crow, pos0, s.vc, R);
        else i2_sconv_static(*q, s.vp, T, KD, l.v_conv, E.cv, crow, pos0, s.vc, R);
        i2_rmsnorm(*q, s.vc, T * NKV, HD, nullptr, c.eps, s.vc, HD, R);
        i2_store_rows(*q, s.vc, T, KD, E.v, pos0, E.ring);
        // 5. relative states (the bias itself is formed inside the attention kernel), 6. sink
        i2_scale_l2clip(*q, s.rb, T * NH, c.d_rel, l.rel_scale, NH, c.rel_clip, s.rb, R);
        if (glob) i2_mul_token(*q, s.rb, T, NH * c.d_rel, s.tau, s.rb, R);
        // 7./8. scores + bias, softmax with the sink, values
        if (!split)                                                     // exact rounding points: the three-pass kernel
            i2_attention(*q, s.qb, E.k, E.v, E.ring, s.rb, l.rel_proj, c.d_rel, l.rel_extent, s.sk, s.o, T, pos0, NH, NKV, HD,
                         glob ? 0 : c.window, R);
        else
            i2_attention_split(*q, s.qb, E.k, E.v, E.ring, s.rb, l.rel_proj, c.d_rel, l.rel_extent, s.sk, s.o, T, pos0, NH,
                               NKV, HD, glob ? 0 : c.window, s.part, kPartFloats, attn_min_keys());
        // 9. gated output (the gate projection ran with the others above)
        i2_mul_sigmoid(*q, s.o, s.gt, uint64_t(T) * QD, s.o, R);
        i2_linear_bf16(*q, s.o, T, QD, l.wo, H, out, R, !R);
    }

    // Returns false when the expert combine was left to the caller's write kernel (fp32 fused path: s.ey, s.w).
    bool moe(uint32_t L, uint32_t T, uint32_t off, uint32_t call, const Inkling2RunOptions& opt, Inkling2Result& res, float* out) {
        const I2Layer& l = layers[L];
        const uint32_t H = c.hidden, K = c.top_k;
        i2_linear_bf16(*q, s.mi, T, H, l.router, c.n_experts, s.logits, false, !R, bm);     // router in fp32
        const int32_t* forced = nullptr;
        if (opt.forced_routes) {
            const auto& fr = (*opt.forced_routes)[call];
            std::vector<int32_t> h(uint64_t(T) * K);
            for (uint32_t t = 0; t < T; ++t)
                for (uint32_t j = 0; j < K; ++j) h[t * K + j] = fr[off + t][j];
            q->memcpy(s.forced, h.data(), h.size() * 4).wait();
            forced = s.forced;
        }
        i2_router(*q, s.logits, l.router_bias, T, c.n_experts, K, c.route_scale, forced, s.used, s.w, s.own, s.margin);
        if (opt.record_routes) {
            q->memcpy(res.own_routes[call].data() + uint64_t(off) * K, s.own, uint64_t(T) * K * 4);
            q->memcpy(res.used_routes[call].data() + uint64_t(off) * K, s.used, uint64_t(T) * K * 4);
            q->memcpy(res.route_margin[call].data() + off, s.margin, uint64_t(T) * 4);
        }
        static const bool no_group = std::getenv("IE_INKLING2_PREFILL_NO_GROUP") != nullptr;   // A/B switch
        static const bool unfused = std::getenv("IE_INKLING2_UNFUSED") != nullptr;
        const bool fuse_act = !R && !unfused && l.w13.cols % 32 == 0;
        // prompt chunks on the matrix engines (IE_INKLING2_EXPERT_XMX=0: the SIMT kernels; _TILE 16 / 32 / 64 slots a
        // tile; _SPLIT=0 plain fp16 inputs: logits 1e-2 off the fp32 path, against 2e-7 with the remainder pass at the
        // same speed). 2K prompt one card: 414 -> 633 tok/s.
        static const int xmx = [] { const char* e = std::getenv("IE_INKLING2_EXPERT_XMX"); return e ? std::atoi(e) : 1; }();
        static const uint32_t xtile = [] { const char* e = std::getenv("IE_INKLING2_EXPERT_XMX_TILE"); const int v = e ? std::atoi(e) : 32; return uint32_t(v == 16 || v == 64 ? v : 32); }();
        static const bool xsplit = [] { const char* e = std::getenv("IE_INKLING2_EXPERT_XMX_SPLIT"); return !e || *e != '0'; }();
        static const bool xmx_mx = [] { const char* e = std::getenv("IE_INKLING2_EXPERT_XMX_MX"); return !e || *e != '0'; }();   // MXFP8 layers too
        static const uint32_t xmin = [] { const char* e = std::getenv("IE_INKLING2_EXPERT_XMX_MINT"); return e ? uint32_t(std::atoi(e)) : 32u; }();
        if (xmx && !R && !no_group && T >= xmin && T > 4 && c.n_experts <= 256 && i2_expert_xmx_ok(l.w13) && i2_expert_xmx_ok(l.w2) &&
            (xmx_mx || (l.w13.fmt == 0 && l.w2.fmt == 0))) {
            const uint32_t maxtiles = uint32_t((uint64_t(T) * K + xtile - 1) / xtile + c.n_experts);
            i2_moe_group(*q, s.used, T, K, c.n_experts, xtile, s.order, s.tiles, s.ntiles);
            i2_expert_xmx(*q, l.w13, s.mi, false, K, s.order, s.tiles, s.ntiles, maxtiles, xtile, xsplit, s.gu);
            i2_expert_act(*q, s.gu, T * K, c.inter, c.gate_softcap, c.up_softcap, s.act, false);
            i2_expert_xmx(*q, l.w2, s.act, true, K, s.order, s.tiles, s.ntiles, maxtiles, xtile, xsplit, s.ey);
            return false;
        }
        if (!R && !no_group && T > 4 && c.n_experts <= 256) {          // prefill: slots grouped by expert
            const uint32_t maxtiles = uint32_t((uint64_t(T) * K + kI2MoeTile - 1) / kI2MoeTile + c.n_experts);
            i2_moe_group(*q, s.used, T, K, c.n_experts, kI2MoeTile, s.order, s.tiles, s.ntiles);
            if (fuse_act) {
                i2_expert_grouped_w13_act(*q, l.w13, s.mi, K, s.order, s.tiles, s.ntiles, maxtiles, s.act, c.gate_softcap, c.up_softcap);
                i2_expert_grouped(*q, l.w2, s.act, true, K, s.order, s.tiles, s.ntiles, maxtiles, s.ey, false);
                return false;                                           // the combine runs inside the layer's write kernel
            }
            i2_expert_grouped(*q, l.w13, s.mi, false, K, s.order, s.tiles, s.ntiles, maxtiles, s.gu, false);
            i2_expert_act(*q, s.gu, T * K, c.inter, c.gate_softcap, c.up_softcap, s.act, false);
            i2_expert_grouped(*q, l.w2, s.act, true, K, s.order, s.tiles, s.ntiles, maxtiles, s.ey, false);
        } else if (fuse_act && tp_n > 1 && T == 1 && !bm && !std::getenv("IE_INKLING2_TP_NOEXPERTS")) {   // (A/B switch)            // tensor parallel: 4 of the 8 picks each
            const uint32_t kh = K / tp_n, j0 = tp_r * kh;
            i2_expert_w13_act(*q, l.w13, s.mi, s.used + j0, 1, kh, s.act + uint64_t(j0) * c.inter, c.gate_softcap, c.up_softcap);
            i2_expert_matvec(*q, l.w2, s.act + uint64_t(j0) * c.inter, true, s.used + j0, 1, kh, s.ey + uint64_t(j0) * H,
                             false, false, true);
            tp_push(peer()->s.ey + uint64_t(j0) * H, s.ey + uint64_t(j0) * H, uint64_t(kh) * H * 4);
            return false;
        } else if (fuse_act) {
            i2_expert_w13_act(*q, l.w13, s.mi, s.used, T, K, s.act, c.gate_softcap, c.up_softcap);
            i2_expert_matvec(*q, l.w2, s.act, true, s.used, T, K, s.ey, false, false, true);
            return false;
        } else {
            i2_expert_matvec(*q, l.w13, s.mi, false, s.used, T, K, s.gu, R, R, !R);
            i2_expert_act(*q, s.gu, T * K, c.inter, c.gate_softcap, c.up_softcap, s.act, R);
            i2_expert_matvec(*q, l.w2, s.act, true, s.used, T, K, s.ey, R, R, !R);
        }
        i2_expert_combine(*q, s.ey, s.w, T, K, H, out, R);
        return true;
    }

    // One layer call (SPEC 4.1-4.4). h: the partial state; BS: block states; msg: [4][T][H] or null.
    void layer(uint32_t L, uint32_t pass, const float* h, const std::vector<const float*>& BS, const float* msg,
               uint32_t T, uint32_t pos0, uint32_t off, uint32_t call, const Inkling2RunOptions& opt, Inkling2Result& res,
               float* out) {
        const I2Layer& l = layers[L];
        const I2Entry& E = entries[c.entry(L, pass)];
        const bool glob = c.is_global(L);
        const uint32_t H = c.hidden;
        const uint64_t TH = uint64_t(T) * H;
        const float *aq = h, *ak = h, *av = h;
        if (L > 0) {
            I2Sources src{};
            for (size_t i = 0; i < BS.size(); ++i) src.p[i] = BS[i];
            src.p[BS.size()] = h;
            i2_attn_res(*q, src, uint32_t(BS.size() + 1), T, H, l.attn_query, glob ? 3 : 1, c.eps, s.A, !R, s.part, bm);
            aq = s.A; ak = glob ? s.A + TH : s.A; av = glob ? s.A + 2 * TH : s.A;
        }
        static const bool unfused = std::getenv("IE_INKLING2_UNFUSED") != nullptr;
        const bool fz = !R && !unfused;
        if (msg && fz) {
            const I2Gate* g3[3] = {&l.ctl[0], &l.ctl[1], &l.ctl[2]};
            const float* h3[3] = {aq, ak, av};
            const float* m3[3] = {msg, msg + TH, msg + 2 * TH};
            float* o3[3] = {s.G[0], s.G[1], s.G[2]};
            gates(3, g3, h3, m3, T, o3);
            aq = s.G[0]; ak = s.G[1]; av = s.G[2];
        } else if (msg) {                                                   // passes 2, 3: gate, then norm (trap 8)
            gate(l.ctl[0], aq, msg + 0 * TH, T, s.G[0]);
            gate(l.ctl[1], ak, msg + 1 * TH, T, s.G[1]);
            gate(l.ctl[2], av, msg + 2 * TH, T, s.G[2]);
            aq = s.G[0]; ak = s.G[1]; av = s.G[2];
        }
        const AblW ga = abl_attn.empty() ? AblW{} : abl_attn[call], gm = abl_moe.empty() ? AblW{} : abl_moe[call];
        const float* aa = add_attn.empty() ? nullptr : add_attn[call];   // steering vectors of this call's writes
        const float* am = add_moe.empty() ? nullptr : add_moe[call];
        const I2Rows brw = bm ? I2Rows{brpos, brslot} : I2Rows{};           // batched: per-row slot / position
        const uint64_t hst = bm ? uint64_t(crow) * H : 0;
        const I2NormSeg ns3[3] = {{aq, l.nq, s.qin}, {ak, l.nk, s.kin}, {av, l.nv, s.vin}};
        i2_rmsnorm_multi(*q, 3, ns3, T, H, c.eps, H, R);
        attention(L, E, T, pos0, s.attn);
        std::vector<const float*> bs = BS;
        const float* residual;
        const bool bnd = c.is_boundary(L);
        if (fz && (ga.k == 0 || ga.k <= 4)) {                           // conv + (ablation) + residual: one launch
            if (bnd) bs.push_back(h);
            if (ga.k) {
                I2WriteAbl w;
                w.moe = false; w.keep_norm = ga.keep_norm; w.T = T; w.H = H; w.hist_rows = crow; w.pos0 = pos0; w.kd = ga.k;
                w.x = s.attn; w.w = l.attn_sconv; w.a = bnd ? nullptr : h; w.dirs = ga.d; w.strength = ga.s;
                w.hist = E.ca; w.y = s.x; w.res = s.res; w.part = s.apart; w.cnt = reinterpret_cast<uint32_t*>(s.acnt);
                w.rows = brw; w.h_stride = hst; w.addv = aa;
                i2_write_ablate(*q, w);
                residual = s.res;
            } else if (bnd && aa) {                                     // boundary + steering: the write itself, plus v
                i2_conv_add(*q, s.attn, T, H, l.attn_sconv, E.ca, crow, pos0, s.x, nullptr, s.res, brw, hst, aa);
                residual = s.res;
            } else {
                i2_conv_add(*q, s.attn, T, H, l.attn_sconv, E.ca, crow, pos0, s.x, bnd ? nullptr : h, bnd ? nullptr : s.res,
                            brw, hst, aa);
                residual = bnd ? s.x : s.res;
            }
            record(call, 1, s.x, T, off);
        } else {
        i2_sconv_static(*q, s.attn, T, H, l.attn_sconv, E.ca, crow, pos0, s.x, R);
        record(call, 1, s.x, T, off);                                   // the attention write, before any ablation
        if (c.is_boundary(L)) {                                         // trap 2: the input moves to the block list
            bs.push_back(h);
            if (ga.k) i2_ablate_add(*q, nullptr, s.x, T, H, ga.d, ga.k, ga.s, s.x, ga.keep_norm);
            residual = s.x;
        } else {
            if (ga.k) i2_ablate_add(*q, h, s.x, T, H, ga.d, ga.k, ga.s, s.res, ga.keep_norm);
            else i2_add(*q, h, s.x, TH, s.res, false);
            residual = s.res;
        }
        if (aa) i2_add_rows(*q, const_cast<float*>(residual), T, H, aa, const_cast<float*>(residual));   // s.x or s.res
        }
        I2Sources src{};
        for (size_t i = 0; i < bs.size(); ++i) src.p[i] = bs[i];
        src.p[bs.size()] = residual;
        i2_attn_res(*q, src, uint32_t(bs.size() + 1), T, H, l.mlp_query, 1, c.eps, s.U, !R, s.part, bm);
        const float* u = s.U;
        if (msg) { gate(l.ctl[3], s.U, msg + 3 * TH, T, s.U2); u = s.U2; }
        record(call, 3, u, T, off);
        i2_rmsnorm(*q, u, T, H, l.mlp_norm, c.eps, s.mi, H, R);
        bool combined = moe(L, T, off, call, opt, res, s.moe);
        if (!combined && T > 4 && !bm) {               // prefill: combine once (the fused tail re-forms it per conv tap)
            i2_expert_combine(*q, s.ey, s.w, T, c.top_k, H, s.moe, false);
            combined = true;
            if (gm.k <= 4) {                           // conv + (ablation) + residual over the combined rows
                if (gm.k) {
                    I2WriteAbl w;
                    w.moe = false; w.keep_norm = gm.keep_norm; w.T = T; w.H = H; w.hist_rows = crow; w.pos0 = pos0;
                    w.kd = gm.k; w.x = s.moe; w.w = l.mlp_sconv; w.a = residual; w.dirs = gm.d; w.strength = gm.s;
                    w.hist = E.cm; w.y = s.y; w.res = out; w.part = s.apart; w.cnt = reinterpret_cast<uint32_t*>(s.acnt);
                    w.addv = am;
                    i2_write_ablate(*q, w);
                } else {
                    i2_conv_add(*q, s.moe, T, H, l.mlp_sconv, E.cm, crow, pos0, s.y, residual, out, {}, 0, am);
                }
                record(call, 2, s.y, T, off);
                record(call, 0, out, T, off);
                return;
            }
        }
        if (!combined && gm.k <= 4) {                  // combine + conv + (ablation) + residual
            if (gm.k) {
                I2WriteAbl w;
                w.moe = true; w.keep_norm = gm.keep_norm; w.T = T; w.H = H; w.k_top = c.top_k; w.hist_rows = crow; w.pos0 = pos0;
                w.kd = gm.k; w.x = s.ey; w.wts = s.w; w.w = l.mlp_sconv; w.a = residual; w.dirs = gm.d; w.strength = gm.s;
                w.hist = E.cm; w.y = s.y; w.res = out; w.part = s.apart; w.cnt = reinterpret_cast<uint32_t*>(s.acnt);
                w.rows = brw; w.h_stride = hst; w.addv = am;
                i2_write_ablate(*q, w);
            } else {
                i2_moe_out(*q, s.ey, s.w, T, c.top_k, H, l.mlp_sconv, E.cm, crow, pos0, s.y, residual, out, brw, hst, am);
            }
            record(call, 2, s.y, T, off);
        } else {
        if (!combined) i2_expert_combine(*q, s.ey, s.w, T, c.top_k, H, s.moe, false);
        i2_sconv_static(*q, s.moe, T, H, l.mlp_sconv, E.cm, crow, pos0, s.y, R);
        record(call, 2, s.y, T, off);
        if (gm.k) i2_ablate_add(*q, residual, s.y, T, H, gm.d, gm.k, gm.s, out, gm.keep_norm);
        else i2_add(*q, residual, s.y, TH, out, R);
        if (am) i2_add_rows(*q, out, T, H, am, out);
        }
        record(call, 0, out, T, off);
    }

    // Recorder: rows of this chunk (global rows off ..) at or after rec_first, into rec [n][calls][kinds][H]
    void record(uint32_t call, uint32_t kind, const float* src, uint32_t T, uint32_t off) {
        if (!rec) return;
        const uint32_t H = c.hidden, NC = c.n_calls();
        for (uint32_t t = 0; t < T; ++t) {
            if (off + t < rec_first) continue;
            const uint64_t r = off + t - rec_first;
            q->memcpy(rec->data() + ((r * NC + call) * kInk2RecKinds + kind) * H, src + uint64_t(t) * H, uint64_t(H) * 4);
        }
    }

    // One chunk of T tokens (rows off .. off + T - 1 of the forward's ids) at positions npos .. npos + T - 1: the 48 calls
    // of SPEC 5.1, then (last chunk only) the head of SPEC 8. Appends to every entry and advances npos.
    std::string chunk(const int32_t* ids, uint32_t T, uint32_t off, const Inkling2RunOptions& opt, Inkling2Result& res,
                      bool last) {
        const uint32_t H = c.hidden, pos0 = bm ? 0 : npos;
        const uint64_t TH = uint64_t(T) * H;
        std::vector<int32_t> qp(T);
        for (uint32_t t = 0; t < T; ++t) qp[t] = bm ? bpos_host[t] : int32_t(pos0 + t);
        q->memcpy(s.qpos, qp.data(), T * 4);
        ink_log_scaling_tau(*q, s.qpos, s.tau, T, c.log_alpha, c.log_n_floor);          // P1's kernel

        // SPEC 2: x0 = rmsnorm(embed[id]; embed_norm), embedding rows gathered on the host
        std::vector<float> e(TH);
        for (uint32_t t = 0; t < T; ++t) {
            const uint8_t* row = embed->data + uint64_t(ids[t]) * H * 2;
            for (uint32_t i = 0; i < H; ++i) { uint16_t b; std::memcpy(&b, row + 2 * i, 2); e[uint64_t(t) * H + i] = i2_bf16_to_f32(b); }
        }
        q->memcpy(s.x0raw, e.data(), e.size() * 4);
        i2_rmsnorm(*q, s.x0raw, T, H, embed_norm, c.eps, s.x0, H, R);
        if (opt.image_rows && opt.image_at && !bm)                      // image features over their placeholders
            for (size_t k = 0; k < opt.image_at->size(); ++k) {
                const uint32_t at = (*opt.image_at)[k];
                if (at >= off && at < off + T) q->memcpy(s.x0 + uint64_t(at - off) * H, opt.image_rows->data() + k * H, uint64_t(H) * 4);
            }
        if (abl_emb.k) i2_ablate_add(*q, nullptr, s.x0, T, H, abl_emb.d, abl_emb.k, abl_emb.s, s.x0, abl_emb.keep_norm);
        if (add_emb) i2_add_rows(*q, s.x0, T, H, add_emb, s.x0);

        uint32_t call = 0;
        const bool feat = opt.features && opt.feature_visits && !bm;
        const uint32_t Tall = feat ? uint32_t(opt.features->size() / std::max<size_t>(1, opt.feature_visits->size() * H)) : 0;
        auto run = [&](uint32_t L, uint32_t pass, const float* h, const std::vector<const float*>& BS, const float* msg) -> const float* {
            if (feat)
                for (size_t k = 0; k < opt.feature_visits->size(); ++k)
                    if ((*opt.feature_visits)[k] == call)
                        q->memcpy(opt.features->data() + (uint64_t(k) * Tall + off) * H, h, TH * 4);
            float* out = s.out[call];
            layer(L, pass, h, BS, msg, T, pos0, off, call, opt, res, out);
            if (opt.keep_calls) q->memcpy(res.calls[call].data() + uint64_t(off) * H, out, TH * 4);
            const float* next = out;
            if (opt.teacher) {                                              // the golden output replaces this call's
                q->memcpy(s.teach[call], (*opt.teacher)[call].data() + uint64_t(off) * H, TH * 4);
                next = s.teach[call];
            }
            ++call;                                                         // no per-call host sync (in-order queue)
            return next;
        };

        // SPEC 5.1
        const float* h = s.x0;
        std::vector<const float*> BS;
        for (uint32_t L = 0; L < c.loop_start; ++L) {
            const float* nh = run(L, 1, h, BS, nullptr);
            if (c.is_boundary(L)) BS.push_back(h);
            h = nh;
        }
        const float* pre_h = h;
        const std::vector<const float*> pre_BS = BS;
        const size_t npre = pre_BS.size();
        for (uint32_t L = c.loop_start; L < c.loop_end; ++L) {              // pass 1
            const float* nh = run(L, 1, h, BS, nullptr);
            if (c.is_boundary(L)) BS.push_back(h);
            h = nh;
        }
        for (uint32_t p = 2; p <= c.passes; ++p) {                          // passes 2..: restart from the pre-loop state
            if (BS.size() != npre + 2) return "internal: unexpected block-state count after a loop pass";
            const float* Cp = BS[npre + 1];                                 // BS[npre + 1:] (trap 5)
            const float* Dp = h;
            h = pre_h;
            BS = pre_BS;
            for (uint32_t L = c.loop_start; L < c.loop_end; ++L) {
                I2Sources src{};
                src.p[0] = Cp; src.p[1] = Dp;
                i2_attn_res(*q, src, 2, T, H, msg_queries[L - c.loop_start], 4, c.eps, s.M, !R, s.part, bm);   // m_q, m_k, m_v, m_mlp
                const float* nh = run(L, p, h, BS, s.M);
                if (c.is_boundary(L)) BS.push_back(h);
                h = nh;
            }
        }
        for (uint32_t L = c.loop_end; L < c.n_layers; ++L) {
            const float* nh = run(L, 1, h, BS, nullptr);
            if (c.is_boundary(L)) BS.push_back(h);
            h = nh;
        }
        if (!bm) { npos = pos0 + T; hw = std::max(hw, npos); }
        if (!last && !feat) { q->wait(); return {}; }                      // the host vectors above (e, qp) die here
        // SPEC 8: final read-out, norm, unembed (last position only), softcap
        I2Sources src{};
        for (size_t i = 0; i < BS.size(); ++i) src.p[i] = BS[i];
        src.p[BS.size()] = h;
        i2_attn_res(*q, src, uint32_t(BS.size() + 1), T, H, output_query, 1, c.eps, s.F, !R, s.part, bm);
        if (feat) {
            for (size_t k = 0; k < opt.feature_visits->size(); ++k)
                if ((*opt.feature_visits)[k] == call) {                     // the final-norm output, every row
                    i2_rmsnorm(*q, s.F, T, H, final_norm, c.eps, s.x0raw, H, false);
                    q->memcpy(opt.features->data() + (uint64_t(k) * Tall + off) * H, s.x0raw, TH * 4);
                }
            if (!last) { q->wait(); return {}; }
        }
        if (bm) {                                                           // batched: every row is a last position
            i2_rmsnorm(*q, s.F, T, H, final_norm, c.eps, bfn, H, false);
            i2_linear_bf16(*q, bfn, T, H, unembed, c.vocab_unpadded, bvocab, false, true, true);
            i2_softcap(*q, bvocab, uint64_t(T) * c.vocab_unpadded, c.final_softcap);
            if (bsamp_on) {                                                 // the draws on the card: T ids come back
                i2_sample_rows(*q, bvocab, T, c.vocab_unpadded, bs.rows, bs.rec, bs.pfi, bs.pfd, bs.cand, bs.out);
                bsamp_ids.resize(T);
                q->memcpy(bsamp_ids.data(), bs.out, uint64_t(T) * 4).wait();
                return {};
            }
            res.last_logits.resize(uint64_t(T) * c.vocab_unpadded);
            q->memcpy(res.last_logits.data(), bvocab, uint64_t(T) * c.vocab_unpadded * 4).wait();
            return {};
        }
        if (opt.all_logits) {                                               // a verify: every row's logits
            const uint64_t need = uint64_t(T) * c.vocab_unpadded;
            if (all_cap < need) {
                if (all_buf) sycl::free(all_buf, *q);
                all_buf = sycl::malloc_device<float>(need, *q);
                if (!all_buf) { all_cap = 0; return "all_logits: device allocation failed"; }
                all_cap = need;
            }
            i2_rmsnorm(*q, s.F, T, H, final_norm, c.eps, s.x0raw, H, false);
            i2_linear_bf16(*q, s.x0raw, T, H, unembed, c.vocab_unpadded, all_buf, false, false);
            i2_softcap(*q, all_buf, need, c.final_softcap);
            res.all_logits.resize(need);
            q->memcpy(res.all_logits.data(), all_buf, need * 4);
        }
        i2_rmsnorm(*q, s.F + uint64_t(T - 1) * H, 1, H, final_norm, c.eps, s.fn, H, R);
        if (tp_n > 1) {                                                 // each rank its half of the vocabulary rows
            const uint32_t V = c.vocab_unpadded, Vh = (V / 2 + 63) / 64 * 64, v0 = tp_r ? Vh : 0, vn = tp_r ? V - Vh : Vh;
            i2_linear_bf16(*q, s.fn, 1, H, unembed + uint64_t(v0) * H, vn, s.vocab + v0, false, true);
            if (tp_r) {                                                 // rank 1: its half to rank 0, which reads out
                tp_push(peer()->s.vocab + v0, s.vocab + v0, uint64_t(vn) * 4);
                q->wait();
                res.last_logits.clear();
                return {};
            }
            tp_sync(q->ext_oneapi_submit_barrier());
        } else
        i2_linear_bf16(*q, s.fn, 1, H, unembed, c.vocab_unpadded, s.vocab, R, !R);
        i2_softcap(*q, s.vocab, c.vocab_unpadded, c.final_softcap);
        res.last_logits.resize(c.vocab_unpadded);
        q->memcpy(res.last_logits.data(), s.vocab, uint64_t(c.vocab_unpadded) * 4).wait();
        return {};
    }
};

Inkling2Model::Inkling2Model() : p_(std::make_unique<Impl>()) {}
Inkling2Model::~Inkling2Model() = default;
const Inkling2Config& Inkling2Model::config() const { return p_->c; }
const std::string& Inkling2Model::ablation_summary() const { return p_->abl_summary; }

std::string Inkling2Model::set_ablation(const Inkling2Ablation* a) {
    Impl& m = *p_;
    if (!m.q) return "not loaded";
    if (m.tp_peer) if (auto e = m.tp_peer->set_ablation(a); !e.empty()) return e;
    m.q->wait();
    m.free_list(m.abl_allocs);
    m.abl_bytes = 0;
    m.abl_attn.clear(); m.abl_moe.clear(); m.abl_emb = {}; m.abl_summary.clear();
    m.add_attn.clear(); m.add_moe.clear(); m.add_emb = nullptr;
    if (!a) return {};
    const uint32_t H = m.c.hidden, NC = m.c.n_calls();
    if (a->dirs.size() != uint64_t(a->rows) * H) return "ablation: directions are not rows x hidden";
    try {
        float* d = m.dev<float>(m.abl_allocs, m.abl_bytes, a->dirs.size());
        m.q->memcpy(d, a->dirs.data(), a->dirs.size() * 4).wait();
        m.abl_attn.assign(NC, {}); m.abl_moe.assign(NC, {});
        m.add_attn.assign(NC, nullptr); m.add_moe.assign(NC, nullptr);
        for (const auto& g : a->groups) {
            if (g.add) {                                                // one precomputed vector per add group
                if (g.k == 0 || uint64_t(g.row0) + g.k > a->rows) return "ablation: malformed group";
                if (g.strength == 0.f) continue;                        // adds nothing: as if absent (bit for bit)
                std::vector<float> v(H, 0.f);
                for (uint32_t r = 0; r < g.k; ++r)
                    for (uint32_t i = 0; i < H; ++i) v[i] += a->dirs[uint64_t(g.row0 + r) * H + i];
                for (auto& x : v) x *= g.strength;
                float* dv = m.dev<float>(m.abl_allocs, m.abl_bytes, H);
                m.q->memcpy(dv, v.data(), uint64_t(H) * 4).wait();
                for (uint32_t i = 0; i < NC; ++i) {
                    if (g.attn[i]) m.add_attn[i] = dv;
                    if (g.moe[i]) m.add_moe[i] = dv;
                }
                if (g.embed) m.add_emb = dv;
                continue;
            }
            if (g.attn.size() != NC || g.moe.size() != NC || g.k == 0 || g.k > kI2AblateMaxK || uint64_t(g.row0) + g.k > a->rows)
                return "ablation: malformed group";
            const Impl::AblW w{d + uint64_t(g.row0) * H, g.k, g.strength, g.keep_norm};
            for (uint32_t i = 0; i < NC; ++i) {
                if (g.attn[i]) m.abl_attn[i] = w;
                if (g.moe[i]) m.abl_moe[i] = w;
            }
            if (g.embed) m.abl_emb = w;
        }
    } catch (const std::exception& e) { return std::string("ablation: ") + e.what(); }
    m.abl_summary = (a->path.empty() ? std::string("(in memory)") : a->path) + ": " + a->summary;
    return {};
}
uint32_t Inkling2Model::n_pos() const { return p_->npos; }
const uint16_t* Inkling2Model::unembed_device() const { return p_->unembed; }
size_t Inkling2Model::device_bytes() const {
    return p_->weight_bytes + p_->fixed_bytes + p_->glob_bytes + p_->scratch_bytes + p_->abl_bytes + p_->batch_bytes;
}

// ---- batched decoding
static uint64_t i2_batch_part_floats(const Inkling2Config& c, uint32_t slots, uint32_t ctx, uint32_t min_keys, uint64_t part1) {
    const uint32_t ns = i2_attention_splits(1, ctx ? ctx - 1 : 0, c.n_heads, c.head_dim, 0, part1, min_keys);
    return uint64_t(slots) * c.n_heads * ns * (c.head_dim + 2);
}

size_t Inkling2Model::batch_slot_bytes(const Inkling2Config& c, uint32_t ctx, uint32_t max_chunk, bool kv_bf16) {
    return state_bytes_for(c, ctx, max_chunk, max_chunk, kv_bf16) + 4 * (uint64_t(c.vocab_unpadded) + c.hidden + 2) +
           4 * i2_batch_part_floats(c, 1, ctx, 128, Impl::kPartFloats);
}

std::string Inkling2Model::batch_begin(uint32_t slots, uint32_t ctx, uint32_t chunk) {
    if (p_->tp_n > 1) return "batch: not with two cards on one conversation (tp_queue)";
    Impl& m = *p_;
    if (!m.q) return "not loaded";
    if (slots == 0 || slots > m.max_chunk) return "batch: slots must be 1.." + std::to_string(m.max_chunk) + " (max_chunk)";
    if (ctx == 0) return "batch: ctx must be > 0";
    if (m.R) return "batch: not in the bf16 emulation mode";
    if (std::getenv("IE_INKLING2_UNFUSED") || Impl::three_pass()) return "batch: needs the fused fp32 path";
    const Inkling2Config& c = m.c;
    if (c.head_dim > 128 || c.n_heads / c.n_kv_heads > 4 || c.d_rel > 16) return "batch: unsupported attention shape";
    batch_end();
    try {
        const uint64_t KD = uint64_t(c.n_kv_heads) * c.head_dim, H = c.hidden, NC = c.n_calls();
        m.kv16 = m.kv16_batch;                                          // kvdev / the slot offsets below
        m.bcap = ctx;
        m.bchunk = std::max<uint32_t>(1, std::min(chunk ? chunk : m.max_chunk, m.max_chunk));
        m.broll = m.bchunk;                                             // a slot rewinds within one prefill chunk
        m.bcrow = m.broll + 3;
        m.bbase.assign(NC, I2Entry{});
        m.bslots.assign(slots, Impl::Slot{});
        for (auto& sl : m.bslots) sl.e.assign(NC, I2Entry{});
        for (uint32_t i = 0; i < NC; ++i) {
            const uint32_t ring = c.is_global(m.entry_layer(i)) ? 0 : c.window + m.broll;
            const uint64_t kvper = uint64_t(ring ? ring : ctx) * KD, cper = uint64_t(m.bcrow) * KD, hper = uint64_t(m.bcrow) * H;
            I2Entry b;
            b.ring = ring;
            b.k = m.kvdev(m.batch_allocs, m.batch_bytes, slots * kvper);
            b.v = m.kvdev(m.batch_allocs, m.batch_bytes, slots * kvper);
            b.ck = m.dev<float>(m.batch_allocs, m.batch_bytes, slots * cper);
            b.cv = m.dev<float>(m.batch_allocs, m.batch_bytes, slots * cper);
            b.ca = m.dev<float>(m.batch_allocs, m.batch_bytes, slots * hper);
            b.cm = m.dev<float>(m.batch_allocs, m.batch_bytes, slots * hper);
            m.bbase[i] = b;
            for (uint32_t s2 = 0; s2 < slots; ++s2) {
                I2Entry& e = m.bslots[s2].e[i];
                e.ring = ring;
                if (m.kv16) {                                           // element offsets of 2-byte elements
                    e.k = reinterpret_cast<float*>(reinterpret_cast<uint16_t*>(b.k) + s2 * kvper);
                    e.v = reinterpret_cast<float*>(reinterpret_cast<uint16_t*>(b.v) + s2 * kvper);
                } else { e.k = b.k + s2 * kvper; e.v = b.v + s2 * kvper; }
                e.ck = b.ck + s2 * cper; e.cv = b.cv + s2 * cper;
                e.ca = b.ca + s2 * hper; e.cm = b.cm + s2 * hper;
            }
        }
        m.bpart = m.dev<float>(m.batch_allocs, m.batch_bytes,
                               i2_batch_part_floats(c, slots, ctx, Impl::attn_min_keys(), Impl::kPartFloats));
        m.bfn = m.dev<float>(m.batch_allocs, m.batch_bytes, uint64_t(slots) * H);
        m.bvocab = m.dev<float>(m.batch_allocs, m.batch_bytes, uint64_t(slots) * c.vocab_unpadded);
        m.brpos = m.dev<int32_t>(m.batch_allocs, m.batch_bytes, slots);
        m.brslot = m.dev<int32_t>(m.batch_allocs, m.batch_bytes, slots);
        m.kv16 = m.kv16_single;
    } catch (const std::exception& e) {
        m.kv16 = m.kv16_single;
        batch_end();
        return std::string("batch: ") + e.what();
    }
    return {};
}

std::string Inkling2Model::batch_step_sample(const std::vector<uint32_t>& slots, const std::vector<int32_t>& ids,
                                             const std::vector<I2SampleRow>& rows, const std::vector<int32_t>& recent,
                                             const std::vector<int32_t>& pf_id, const std::vector<float>& pf_d,
                                             std::vector<int32_t>& out) {
    Impl& m = *p_;
    const uint32_t n = uint32_t(slots.size());
    if (rows.size() != n || pf_id.size() != pf_d.size()) return "batch_step_sample: one parameter row per slot";
    try {
        auto grow = [&](auto*& p, uint32_t& cap, size_t need) {
            using T = std::remove_reference_t<decltype(*p)>;
            if (need <= cap && p) return;
            if (p) sycl::free(p, *m.q);
            cap = uint32_t(std::max<size_t>(need, 64));
            p = sycl::malloc_device<T>(cap, *m.q);
            if (!p) throw std::runtime_error("device allocation failed (sampling buffers)");
        };
        if (n > m.bs.nrows || !m.bs.cand) {
            m.q->wait();
            if (m.bs.cand) sycl::free(m.bs.cand, *m.q);
            if (m.bs.out) sycl::free(m.bs.out, *m.q);
            if (m.bs.rows) sycl::free(m.bs.rows, *m.q);
            const uint32_t cap = std::max<uint32_t>(n, uint32_t(m.bslots.size()));
            m.bs.cand = sycl::malloc_device<int32_t>(uint64_t(cap) * m.c.vocab_unpadded, *m.q);
            m.bs.out = sycl::malloc_device<int32_t>(cap, *m.q);
            m.bs.rows = sycl::malloc_device<I2SampleRow>(cap, *m.q);
            if (!m.bs.cand || !m.bs.out || !m.bs.rows) return "batch_step_sample: device allocation failed (sampling buffers)";
            m.bs.nrows = cap;
        }
        if (!recent.empty()) { if (recent.size() > m.bs.nrec || !m.bs.rec) { m.q->wait(); grow(m.bs.rec, m.bs.nrec, recent.size()); } }
        if (!pf_id.empty()) {
            if (pf_id.size() > m.bs.npf || !m.bs.pfi) {
                m.q->wait();
                uint32_t c2 = m.bs.npf;
                grow(m.bs.pfi, m.bs.npf, pf_id.size());
                grow(m.bs.pfd, c2, pf_id.size());
            }
        }
        m.q->memcpy(m.bs.rows, rows.data(), uint64_t(n) * sizeof(I2SampleRow));
        if (!recent.empty()) m.q->memcpy(m.bs.rec, recent.data(), recent.size() * 4);
        if (!pf_id.empty()) { m.q->memcpy(m.bs.pfi, pf_id.data(), pf_id.size() * 4); m.q->memcpy(m.bs.pfd, pf_d.data(), pf_d.size() * 4); }
        m.q->wait();
    } catch (const std::exception& e) { return std::string("batch_step_sample: ") + e.what(); }
    m.bsamp_on = true;
    std::vector<float> none;
    const std::string e = batch_step(slots, ids, none);
    m.bsamp_on = false;
    if (!e.empty()) return e;
    out = m.bsamp_ids;
    return {};
}

void Inkling2Model::batch_end() {
    Impl& m = *p_;
    if (m.q) m.q->wait();
    m.bs_free();
    m.free_list(m.batch_allocs);
    m.batch_bytes = 0;
    m.bslots.clear(); m.bbase.clear();
    m.bcap = 0;
    m.bpart = m.bfn = m.bvocab = nullptr;
    m.brpos = m.brslot = nullptr;
}

uint32_t Inkling2Model::batch_slots() const { return uint32_t(p_->bslots.size()); }
uint32_t Inkling2Model::batch_ctx() const { return p_->bcap; }

std::string Inkling2Model::batch_fill_zero_for_benchmark(uint32_t slot, uint32_t n) {
    Impl& m = *p_;
    if (slot >= m.bslots.size()) return "batch: no slot " + std::to_string(slot);
    if (n >= m.bcap) return "batch: " + std::to_string(n) + " positions do not fit the slots' context";
    const uint64_t KD = uint64_t(m.c.n_kv_heads) * m.c.head_dim, H = m.c.hidden;
    for (const auto& e : m.bslots[slot].e) {
        const uint64_t rows = e.ring ? e.ring : n;
        const uint64_t eb = m.kv16_batch ? 2 : 4;
        m.q->memset(e.k, 0, rows * KD * eb); m.q->memset(e.v, 0, rows * KD * eb);
        m.q->memset(e.ck, 0, m.bcrow * KD * 4); m.q->memset(e.cv, 0, m.bcrow * KD * 4);
        m.q->memset(e.ca, 0, m.bcrow * H * 4);  m.q->memset(e.cm, 0, m.bcrow * H * 4);
    }
    m.q->wait();
    m.bslots[slot].npos = m.bslots[slot].hw = n;
    return {};
}

uint32_t Inkling2Model::batch_pos(uint32_t slot) const {
    return slot < p_->bslots.size() ? p_->bslots[slot].npos : 0;
}

uint32_t Inkling2Model::batch_earliest_restorable(uint32_t slot) const {
    const Impl& m = *p_;
    if (slot >= m.bslots.size()) return 0;
    const uint32_t hw = m.bslots[slot].hw;
    return hw > m.broll ? hw - m.broll : 0;
}

std::string Inkling2Model::batch_truncate(uint32_t slot, uint32_t n) {
    Impl& m = *p_;
    if (slot >= m.bslots.size()) return "batch: no slot " + std::to_string(slot);
    auto& b = m.bslots[slot];
    if (n == 0) { b.npos = b.hw = 0; return {}; }
    if (n > b.npos) return "batch_truncate: " + std::to_string(n) + " is beyond the slot's " + std::to_string(b.npos) + " positions";
    if (n < batch_earliest_restorable(slot))
        return "batch_truncate: position " + std::to_string(n) + " is no longer restorable (earliest " +
               std::to_string(batch_earliest_restorable(slot)) + ")";
    b.npos = n;
    return {};
}

std::string Inkling2Model::batch_forward(uint32_t slot, const std::vector<int32_t>& ids, const Inkling2RunOptions& opt,
                                         Inkling2Result& out) {
    Impl& m = *p_;
    if (slot >= m.bslots.size()) return "batch: no slot " + std::to_string(slot);
    if (uint64_t(m.bslots[slot].npos) + ids.size() > m.bcap)
        return "batch: " + std::to_string(m.bslots[slot].npos + ids.size()) + " positions exceed the slots' context " + std::to_string(m.bcap);
    // the single state's fields, put back below; the slot's rings are window + chunk rows: its chunks are at most that
    const uint32_t npos0 = m.npos, hw0 = m.hw, cap0 = m.cap, crow0 = m.crow, roll0 = m.rollback;
    std::swap(m.entries, m.bslots[slot].e);
    m.npos = m.bslots[slot].npos; m.hw = m.bslots[slot].hw; m.cap = m.bcap; m.crow = m.bcrow; m.rollback = m.broll;
    m.kv16 = m.kv16_batch;
    Inkling2RunOptions o = opt;
    o.chunk = std::min(opt.chunk ? opt.chunk : m.bchunk, m.bchunk);
    std::string e;
    try { e = forward(ids, o, out); } catch (...) { e = "batch_forward: exception"; }
    m.kv16 = m.kv16_single;
    m.bslots[slot].npos = m.npos; m.bslots[slot].hw = m.hw;
    std::swap(m.entries, m.bslots[slot].e);
    m.npos = npos0; m.hw = hw0; m.cap = cap0; m.crow = crow0; m.rollback = roll0;
    if (!e.empty()) m.bslots[slot].npos = m.bslots[slot].hw = 0;
    return e;
}

std::string Inkling2Model::batch_prefill(uint32_t slot, const std::vector<int32_t>& ids, const Inkling2RunOptions& opt,
                                         Inkling2Result& out) {
    Impl& m = *p_;
    if (slot >= m.bslots.size()) return "batch: no slot " + std::to_string(slot);
    m.bslots[slot].npos = m.bslots[slot].hw = 0;                        // a fresh conversation
    return batch_forward(slot, ids, opt, out);
}

std::string Inkling2Model::batch_step(const std::vector<uint32_t>& slots, const std::vector<int32_t>& ids,
                                      std::vector<float>& logits) {
    Impl& m = *p_;
    const Inkling2Config& c = m.c;
    const uint32_t n = uint32_t(slots.size());
    if (m.bslots.empty()) return "batch: batch_begin first";
    if (n == 0 || ids.size() != n) return "batch: need one id per slot";
    if (n > m.max_chunk) return "batch: more rows than max_chunk";
    std::vector<int32_t> pos(n), sl(n);
    std::vector<char> seen(m.bslots.size(), 0);
    for (uint32_t i = 0; i < n; ++i) {
        if (slots[i] >= m.bslots.size()) return "batch: no slot " + std::to_string(slots[i]);
        if (seen[slots[i]]++) return "batch: slot " + std::to_string(slots[i]) + " listed twice";
        if (ids[i] < 0 || uint32_t(ids[i]) >= c.vocab) return "token id out of range: " + std::to_string(ids[i]);
        if (m.bslots[slots[i]].npos + 1 > m.bcap) return "batch: slot " + std::to_string(slots[i]) + " is full";
        pos[i] = int32_t(m.bslots[slots[i]].npos);
        sl[i] = int32_t(slots[i]);
    }
    for (size_t i = 0; i < m.abl_attn.size(); ++i)
        if (m.abl_attn[i].k > 4 || m.abl_moe[i].k > 4) return "batch: ablation groups of more than 4 directions are not supported";
    try {
        m.q->memcpy(m.brpos, pos.data(), uint64_t(n) * 4);
        m.q->memcpy(m.brslot, sl.data(), uint64_t(n) * 4).wait();
        m.bpos_host = pos;
        m.bmaxpos = uint32_t(*std::max_element(pos.begin(), pos.end()));
        std::swap(m.entries, m.bbase);                                  // slot 0's pointers + per-row strides
        const uint32_t crow0 = m.crow, roll0 = m.rollback;
        m.crow = m.bcrow; m.rollback = m.broll; m.kv16 = m.kv16_batch;
        m.bm = true;
        m.rec = nullptr;
        Inkling2RunOptions ro;
        ro.record_routes = false;
        Inkling2Result r;
        std::string e;
        try { e = m.chunk(ids.data(), n, 0, ro, r, true); }
        catch (...) { m.bm = false; std::swap(m.entries, m.bbase); m.crow = crow0; m.rollback = roll0; m.kv16 = m.kv16_single; throw; }
        m.bm = false;
        std::swap(m.entries, m.bbase);
        m.crow = crow0; m.rollback = roll0; m.kv16 = m.kv16_single;
        if (!e.empty()) return e;
        logits = std::move(r.last_logits);
    } catch (const sycl::exception& e) {
        return std::string("inkling2 batch_step: SYCL: ") + e.what();
    } catch (const std::exception& e) {
        return std::string("inkling2 batch_step: ") + e.what();
    }
    for (uint32_t i = 0; i < n; ++i) {
        auto& b = m.bslots[slots[i]];
        ++b.npos;
        b.hw = std::max(b.hw, b.npos);
    }
    return {};
}

size_t Inkling2Model::state_bytes_for(const Inkling2Config& c, uint32_t ctx, uint32_t max_chunk) {
    return state_bytes_for(c, ctx, max_chunk, max_chunk);
}

bool Inkling2Model::kv_bf16() const { return p_->kv16_single; }
bool Inkling2Model::batch_kv_bf16() const { return p_->kv16_batch; }
std::string Inkling2Model::kv_note() const {
    return std::string("keys / values ") + (p_->kv16_single ? "bf16" : "fp32") + " (one conversation), " +
           (p_->kv16_batch ? "bf16" : "fp32") + " (batch slots): " + p_->kv_why;
}

size_t Inkling2Model::state_bytes_for(const Inkling2Config& c, uint32_t ctx, uint32_t max_chunk, uint32_t rollback, bool kv_bf16) {
    rollback = std::max(rollback, max_chunk);
    const uint64_t KD = uint64_t(c.n_kv_heads) * c.head_dim, H = c.hidden;
    uint64_t n_glob = 0;
    for (uint32_t i = 0; i < c.n_calls(); ++i) {
        const uint32_t L = i < c.n_layers ? i : c.loop_start + (i - c.n_layers) % c.n_loop();
        n_glob += c.is_global(L);
    }
    const uint64_t n_slide = c.n_calls() - n_glob;
    return size_t((kv_bf16 ? 2 : 4) * 2 * KD * (n_glob * ctx + n_slide * (uint64_t(c.window) + rollback)) +
                  4 * uint64_t(c.n_calls()) * (rollback + 3) * (2 * KD + 2 * H));
}

size_t Inkling2Model::scratch_bytes_for(const Inkling2Config& c, uint32_t max_chunk, bool teacher) {
    Impl m;
    m.c = c;
    return m.alloc_scratch(max_chunk, teacher, true);
}

std::string Inkling2Model::load(const std::string& dir, sycl::queue& q, const Inkling2LoadOptions& opt) {
    Impl& m = *p_;
    if (m.q) return "already loaded";
    if (auto e = inkling2_read_config(dir, m.c); !e.empty()) return e;
    if (opt.max_chunk == 0 || opt.max_chunk > 4096) return "max_chunk must be 1..4096";
    if (auto e = m.st.open(dir); !e.empty()) return e;
    m.q = &q;
    m.R = opt.ref_bf16;
    if (opt.tp_queue) {                                         // two cards on one conversation: this is rank 0
        const Inkling2Config& c = m.c;
        if (opt.tp_queue->get_context() != q.get_context()) return "tp: the two queues must share one context";
        if (opt.tp_queue->get_device() == q.get_device()) return "tp: the two queues are on the same card";
        if (m.R || std::getenv("IE_INKLING2_UNFUSED") || Impl::three_pass() || c.head_dim > 128 || c.n_heads / c.n_kv_heads > 4 ||
            c.n_heads % 2 || c.n_kv_heads % 2 || c.hidden % 2 || c.top_k % 2)
            return "tp: needs the fused fp32 path and even head / expert counts";
        try {
            auto d0 = q.get_device(), d1 = opt.tp_queue->get_device();
            if (!d0.ext_oneapi_can_access_peer(d1, sycl::ext::oneapi::peer_access::access_supported) ||
                !d1.ext_oneapi_can_access_peer(d0, sycl::ext::oneapi::peer_access::access_supported))
                return "tp: no peer access between the two cards";
            d0.ext_oneapi_enable_peer_access(d1);
            d1.ext_oneapi_enable_peer_access(d0);
        } catch (const sycl::exception& e) {
            if (std::string(e.what()).find("already") == std::string::npos) return std::string("tp: peer access: ") + e.what();
        }
        m.tp_n = 2;
        m.tp_r = 0;
    }
    m.max_chunk = opt.max_chunk;
    m.rollback = std::max(opt.rollback, opt.max_chunk);
    m.crow = m.rollback + 3;
    {   // bf16 keys / values: the fused fp32 path only (its prep stores and split attention read them)
        const char* kvs = std::getenv("IE_INKLING2_KV");
        const std::string ks = kvs ? kvs : "";
        const Inkling2Config& c = m.c;
        const bool can = !m.R && !std::getenv("IE_INKLING2_UNFUSED") && !Impl::three_pass() && c.head_dim <= 128 &&
                         c.n_heads / c.n_kv_heads <= 4 && c.d_rel <= 16;
        const bool fs = ks == "fp32", bs = ks == "bf16";
        m.kv16_single = can && (bs || (!fs && opt.kv_bf16));
        m.kv16_batch = can && (bs || (!fs && opt.kv_bf16_batch));
        m.kv16 = m.kv16_single;
        m.kv_why = !can ? "the bf16 emulation / unfused path keeps fp32"
                 : fs || bs ? "IE_INKLING2_KV=" + ks
                 : "defaults: one conversation fp32 (closest to the reference at long context), batch slots bf16 (half the "
                   "bytes per slot); IE_INKLING2_KV=fp32 / bf16 sets both";
    }
    try {
        m.load_all(dir);
        m.init_state();
        m.alloc_scratch(m.max_chunk, false, false);
    } catch (const std::exception& e) { return std::string("inkling2 load: ") + e.what(); }
    if (opt.tp_queue) {                                         // rank 1: the same load on the other card, then the link
        m.tp_peer = std::make_unique<Inkling2Model>();
        m.tp_peer->p_->tp_n = 2;
        m.tp_peer->p_->tp_r = 1;
        Inkling2LoadOptions o1 = opt;
        o1.tp_queue = nullptr;
        o1.ablation = "off";                                    // set_ablation below sets both
        if (auto e = m.tp_peer->load(dir, *opt.tp_queue, o1); !e.empty()) { m.tp_peer.reset(); return "tp (second card): " + e; }
        m.tp = std::make_shared<I2TpLink>();
        m.tp->m[0] = &m;
        m.tp->m[1] = m.tp_peer->p_.get();
        m.tp_peer->p_->tp = m.tp;
        m.tp_thr = std::thread([&m] { m.tp_worker(); });
    }
    // the ablation set: the option, else IE_INKLING2_ABLATION, else <dir>/ie_ablation.json if present; "off" = none
    std::string ap = opt.ablation;
    if (ap.empty()) if (const char* e = std::getenv("IE_INKLING2_ABLATION"); e && *e) ap = e;
    if (ap.empty()) {
        const std::string d = dir + "/ie_ablation.json";
        if (std::ifstream(d).good()) ap = d;
    }
    if (!ap.empty() && ap != "off") {
        Inkling2Ablation a;
        if (auto e = inkling2_read_ablation(ap, m.c, a); !e.empty()) return e;
        if (auto e = set_ablation(&a); !e.empty()) return e;
    }
    return {};
}

void Inkling2Model::reset_state() {
    Impl& m = *p_;
    if (!m.q) return;
    if (m.tp_peer) m.tp_peer->reset_state();
    m.npos = 0;
    m.hw = 0;                                                   // positions < 0 read as zero: nothing to clear
}

uint32_t Inkling2Model::earliest_restorable() const {
    const Impl& m = *p_;
    return m.hw > m.rollback ? m.hw - m.rollback : 0;
}

std::string Inkling2Model::truncate(uint32_t n) {
    Impl& m = *p_;
    if (!m.q) return "not loaded";
    if (n == 0) { reset_state(); return {}; }
    if (m.tp_peer) if (auto e = m.tp_peer->truncate(n); !e.empty()) return e;
    if (n > m.npos) return "truncate: " + std::to_string(n) + " is beyond the cached " + std::to_string(m.npos) + " positions";
    if (n < earliest_restorable())
        return "truncate: position " + std::to_string(n) + " is no longer restorable (earliest " +
               std::to_string(earliest_restorable()) + "; rollback " + std::to_string(m.rollback) + ")";
    m.npos = n;
    return {};
}

std::string Inkling2Model::fill_zero_context_for_benchmark(uint32_t n) {
    Impl& m = *p_;
    if (!m.q) return "not loaded";
    if (m.tp_peer) if (auto e = m.tp_peer->fill_zero_context_for_benchmark(n); !e.empty()) return e;
    try {
        if (n > m.cap) m.grow_global(n);
        const uint64_t KD = m.kd_l(), H = m.c.hidden;
        for (auto& e : m.entries) {
            const uint64_t rows = e.ring ? e.ring : n;
            m.q->memset(e.k, 0, rows * KD * m.kvb()); m.q->memset(e.v, 0, rows * KD * m.kvb());
            m.q->memset(e.ck, 0, m.crow * KD * 4); m.q->memset(e.cv, 0, m.crow * KD * 4);
            m.q->memset(e.ca, 0, m.crow * H * 4);  m.q->memset(e.cm, 0, m.crow * H * 4);
        }
        m.q->wait();
    } catch (const std::exception& e) { return std::string("inkling2 fill: ") + e.what(); }
    m.npos = n;
    m.hw = n;
    return {};
}

std::string Inkling2Model::decode_step(int32_t id, std::vector<float>& logits) {
    Inkling2RunOptions ro;
    ro.record_routes = false;
    Inkling2Result r;
    if (auto e = forward({id}, ro, r); !e.empty()) return e;
    logits = std::move(r.last_logits);
    return {};
}

std::string Inkling2Model::reserve(uint32_t ctx) {
    Impl& m = *p_;
    if (!m.q) return "not loaded";
    if (m.tp_peer) if (auto e = m.tp_peer->reserve(ctx); !e.empty()) return e;
    try { if (ctx > m.cap) m.grow_global(ctx); } catch (const std::exception& e) { return std::string("inkling2 reserve: ") + e.what(); }
    return {};
}

std::string Inkling2Model::forward(const std::vector<int32_t>& ids, const Inkling2RunOptions& opt, Inkling2Result& res) {
    Impl& m = *p_;
    if (!m.q) return "not loaded";
    if (m.tp_peer && !m.tp_in) {                    // two cards: the other card's half on the worker, this one's here
        Inkling2RunOptions o1 = opt;                // host read-backs from rank 0 only
        o1.record_out = nullptr; o1.record_last = 0; o1.keep_calls = false; o1.record_routes = false;
        o1.features = nullptr; o1.feature_visits = nullptr;
        Inkling2Result r1;
        std::string e1;
        m.tp->abort.store(false);
        m.tp_post([&] { e1 = m.tp_peer->forward(ids, o1, r1); if (!e1.empty()) m.tp->abort.store(true); });
        m.tp_in = true;
        std::string e0 = forward(ids, opt, res);
        m.tp_in = false;
        if (!e0.empty()) m.tp->abort.store(true);
        m.tp_wait();
        if (e0.empty() && e1.empty()) {
            const uint32_t g0 = m.tp_gave_up(), g1 = m.tp_peer->p_->tp_gave_up();
            if (g0 || g1) e0 = "inkling2 two cards: a card-to-card exchange never arrived (exchange " + std::to_string(g0 ? g0 : g1) + ")";
        }
        if (!e0.empty() || !e1.empty()) {           // the ranks' exchange counts may now differ: start both over
            m.tp->n[0] = m.tp->n[1] = 0;
            for (auto& r : m.tp->ring) for (auto& sl : r) sl.seq.store(0);
            m.tp_reset_flags(); m.tp_peer->p_->tp_reset_flags();
            m.npos = m.hw = 0; m.tp_peer->p_->npos = m.tp_peer->p_->hw = 0;
            return !e0.empty() ? e0 : "inkling2 (second card): " + e1;
        }
        return {};
    }
    const Inkling2Config& c = m.c;
    const uint32_t Ttot = uint32_t(ids.size()), H = c.hidden, NC = c.n_calls(), K = c.top_k;
    if (Ttot == 0) return "no ids";
    for (int32_t id : ids) if (id < 0 || uint32_t(id) >= c.vocab) return "token id out of range: " + std::to_string(id);
    if (opt.forced_routes) {
        if (opt.forced_routes->size() != NC) return "forced routes: need one entry per call";
        for (const auto& cr : *opt.forced_routes) {
            if (cr.size() != Ttot) return "forced routes: need one row per token";
            for (const auto& r : cr) for (int32_t e : r) if (e < 0 || uint32_t(e) >= c.n_experts) return "forced routes: bad expert id";
        }
    }
    if (opt.teacher) {
        if (opt.teacher->size() != NC) return "teacher: need one array per call";
        for (const auto& a : *opt.teacher) if (a.size() != uint64_t(Ttot) * H) return "teacher: need [T, hidden] per call";
    }
    if (opt.image_rows || opt.image_at) {
        if (!opt.image_rows || !opt.image_at || opt.image_rows->size() != opt.image_at->size() * uint64_t(H))
            return "images: need image_rows [n, hidden] and image_at [n]";
        for (uint32_t at : *opt.image_at) if (at >= Ttot) return "images: a position is beyond this forward's ids";
    }
    const uint32_t C = opt.chunk ? std::min(opt.chunk, m.max_chunk) : m.max_chunk;
    try {
        if (m.npos + Ttot > m.cap) m.grow_global(std::max(m.npos + Ttot, m.cap ? 2 * m.cap : 1024u));
        if (opt.teacher && !m.scratch_teacher) m.alloc_scratch(m.max_chunk, true, false);
        res = Inkling2Result{};
        m.rec = nullptr;
        if (opt.record_last && opt.record_out) {
            if (opt.record_last > Ttot) return "record_last exceeds the forward's tokens";
            opt.record_out->assign(uint64_t(opt.record_last) * NC * kInk2RecKinds * H, 0.f);
            m.rec = opt.record_out;
            m.rec_first = Ttot - opt.record_last;
        }
        res.own_routes.assign(NC, std::vector<int32_t>(opt.record_routes ? uint64_t(Ttot) * K : 0));
        res.used_routes.assign(NC, std::vector<int32_t>(opt.record_routes ? uint64_t(Ttot) * K : 0));
        res.route_margin.assign(NC, std::vector<float>(opt.record_routes ? Ttot : 0));
        if (opt.keep_calls) res.calls.assign(NC, std::vector<float>(uint64_t(Ttot) * H));
        if (opt.all_logits && (Ttot > C || m.tp_n > 1)) return "all_logits: the forward must fit one chunk, on one card";
        if (opt.features && opt.feature_visits) {
            for (uint32_t v : *opt.feature_visits) if (v > NC) return "features: a visit beyond the calls";
            opt.features->assign(opt.feature_visits->size() * uint64_t(Ttot) * H, 0.f);
        }
        for (uint32_t off = 0; off < Ttot; off += C) {
            const uint32_t T = std::min(C, Ttot - off);
            if (auto e = m.chunk(ids.data() + off, T, off, opt, res, off + T == Ttot); !e.empty()) { m.rec = nullptr; return e; }
        }
        m.rec = nullptr;
    } catch (const sycl::exception& e) {
        return std::string("inkling2 forward: SYCL: ") + e.what();
    } catch (const std::exception& e) {
        return std::string("inkling2 forward: ") + e.what();
    }
    return {};
}

}  // namespace ie
