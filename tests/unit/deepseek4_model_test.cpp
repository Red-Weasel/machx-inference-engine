// tests/unit/deepseek4_model_test.cpp — DeepSeek4Model::load() against the REAL
// shipped GGUFs of BOTH quants: UD-Q3_K_XL (4 shards) and UD-Q8_K_XL (5 shards).
//
// WHAT DEFECT THIS TEST EXISTS FOR
// --------------------------------
// Phase 1 was gate-passed on the principle that dtype dispatch reads each
// tensor's own `type_id` and never a per-model or per-role constant. That
// discipline was applied to the routed experts (which is why blk.26's MXFP4
// gate/up bind alongside its IQ3_XXS siblings) and NOT to the dense set: the
// loader hard-coded Q8_0 for every attention projection, shared expert and
// `token_embd`, Q6_K for `output`, and F32 for `indexer.proj`. UD-Q8_K_XL stores
// every one of those as BF16, so it could not bind a single layer — the whole
// 161.86 GB / 1328-tensor model was unloadable, and every Q8 figure in the docs
// was arithmetic over a tensor table rather than a load.
//
// So the test runs the SAME check function over both quants. One code path must
// serve both, and the traps are asserted by name and dtype rather than by count:
//   Q8 `attn_q_b` is BF16, NOT Q8_0        Q8 `output` is BF16, NOT Q6_K
//   Q8 `token_embd` is BF16, NOT Q8_0      Q8 `indexer.proj` is BF16, NOT F32
//   Q3 blk.26 gate/up are MXFP4 while blk.25/blk.27 are IQ3_XXS
//   Q8 gate/up are MXFP4 on ALL 43 layers, so blk.26 is not special there
//
// Opens shard 1 of each; GgufReader pulls in the siblings itself and merges the
// tensor tables, so this exercises the multi-shard path at both 4 and 5 shards.
// mmap + header parse only — no tensor bytes are touched, no GPU. Skips-with-
// warning (exit 0) per model if that model is absent, mirroring
// tests/unit/deepseek4_config_test.cpp so CI on other boxes stays green.
#undef NDEBUG  // build is Release (-DNDEBUG); asserts must stay live here
#include "ie/deepseek4.hpp"
#include "ie/expert_stream.hpp"
#include "ie/model_config.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

namespace {

constexpr const char* kQ3Shard1 =
    "${IE_MODELS_DIR}/DeepSeek-V4-Flash-0731-GGUF/UD-Q3_K_XL/"
    "DeepSeek-V4-Flash-0731-UD-Q3_K_XL-00001-of-00004.gguf";
constexpr const char* kQ8Shard1 =
    "${IE_MODELS_DIR}/DeepSeek-V4-Flash-0731-GGUF/UD-Q8_K_XL/"
    "DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf";

int g_fail = 0;

void ok(const std::string& what, bool cond, const std::string& detail = {}) {
    std::printf("  [%s] %s%s%s\n", cond ? "PASS" : "FAIL", what.c_str(),
                detail.empty() ? "" : " -- ", detail.c_str());
    if (!cond) ++g_fail;
}

std::string env_or(const char* key, const char* fallback) {
    const char* v = std::getenv(key);
    return v ? std::string(v) : std::string(fallback);
}

std::string gb(uint64_t b) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.3f GB (%llu B)", double(b) / 1e9, (unsigned long long)b);
    return buf;
}

bool has(const ie::GgufReader& g, uint32_t L, const char* suffix) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "blk.%u.%s", L, suffix);
    return g.find_tensor(buf) != nullptr;
}

std::string tn(const ie::GgufTensorInfo* t) {
    return t ? std::string(ie::type_name(t->dtype)) : std::string("<null>");
}

// What one bound model looks like, so the caller can compare the two quants.
struct Bound {
    bool        opened = false;
    std::string label;
    uint64_t    n_bound = 0;
    // Resident-set bytes charged the way DeepSeek4Runtime::load charges them,
    // walked over the POINTERS load() bound rather than over names in the file.
    uint64_t    resident_device = 0;   // what upload_* actually allocates
    uint64_t    resident_gguf   = 0;   // what the file stores for that same set
    uint64_t    n_packed = 0, n_expanded = 0;
    // Routed experts, for the pinned-host figure.
    std::vector<uint64_t> slot_bytes;
    uint64_t    per_layer = 0, pool = 0;
};

// -------------------------------------------------------------------------
// The whole Phase-1 binding contract, run identically on either quant.
//
// The RETURN VALUE means "the file is present", NOT "the checks passed" — a
// model that is present but fails to bind must make the test EXIT NONZERO, not
// fall through to the skip path. `out.opened` is the separate "fully validated"
// flag that gates everything downstream.
// -------------------------------------------------------------------------
bool check_model(const char* label, const std::string& shard1, Bound& out) {
    out.label = label;
    ie::GgufReader g;
    const std::string err = g.open(shard1);
    if (!err.empty()) {
        std::printf("  SKIP %s: cannot open %s (%s)\n", label, shard1.c_str(), err.c_str());
        return false;   // absent — the ONLY reason this test may skip
    }
    std::printf("\n  ---- %s ----\n", label);

    ie::DeepSeek4Config cfg;
    const std::string cerr = ie::read_deepseek4_config(g, cfg);
    ok(std::string(label) + ": config reads", cerr.empty(), cerr);
    if (!cerr.empty()) return true;

    ie::DeepSeek4Model m;
    const std::string lerr = m.load(g, cfg);
    // THE BLOCKER, stated as an assertion: before the per-tensor dense dtype
    // this returned `blk.0.attn_q_a.weight: expected dtype Q8_0, file has BF16`
    // for Q8 and bound nothing.
    ok(std::string(label) + ": load() succeeds", lerr.empty(), lerr);
    if (!lerr.empty()) return true;

    // -- every tensor in the file is bound, and nothing else ----------------
    ok(std::string(label) + ": all 1328 tensors bound", m.n_bound() == 1328 &&
       g.n_tensors() == 1328,
       std::to_string(m.n_bound()) + " bound of " + std::to_string(g.n_tensors()) + " in file");
    ok(std::string(label) + ": 43 layers", m.layers().size() == 43);
    out.n_bound = m.n_bound();

    for (const auto& w : m.layers()) {
        assert(w.attn_norm && w.attn_q_a && w.attn_q_a_norm && w.attn_q_b);
        assert(w.attn_kv && w.attn_kv_a_norm && w.attn_sinks);
        assert(w.attn_output_a && w.attn_output_b);
        assert(w.ffn_norm && w.ffn_gate_inp);
        assert(w.ffn_gate_exps && w.ffn_up_exps && w.ffn_down_exps);
        assert(w.ffn_gate_shexp && w.ffn_up_shexp && w.ffn_down_shexp);
        assert(w.hc_attn_fn && w.hc_attn_base && w.hc_attn_scale);
        assert(w.hc_ffn_fn && w.hc_ffn_base && w.hc_ffn_scale);
    }
    const auto& gl = m.globals();
    assert(gl.token_embd && gl.output && gl.output_norm);
    assert(gl.output_hc_fn && gl.output_hc_base && gl.output_hc_scale);
    assert(gl.output_hc_fn->shape[0] == 16384 && gl.output_hc_fn->shape[1] == 4);

    // -- per-layer variation, cross-checked against the FILE -----------------
    int n_hash = 0, n_compressor = 0, n_indexer = 0;
    for (uint32_t L = 0; L < 43; ++L) {
        const auto& w = m.layers()[L];

        assert(w.kind.hash_router == (L < 3));
        assert(w.kind.hash_router == has(g, L, "ffn_gate_tid2eid.weight"));
        assert(w.kind.hash_router != has(g, L, "exp_probs_b.bias"));
        assert((w.ffn_gate_tid2eid != nullptr) == w.kind.hash_router);
        assert((w.exp_probs_b != nullptr) != w.kind.hash_router);
        if (w.kind.hash_router) {
            assert(w.ffn_gate_tid2eid->dtype == ie::DType::kI32);
            assert(w.ffn_gate_tid2eid->shape[0] == 6);
            assert(w.ffn_gate_tid2eid->shape[1] == 129280);
            ++n_hash;
        } else {
            assert(w.exp_probs_b->shape[0] == 256);
            assert(w.exp_probs_b->dtype == ie::DType::kF32);
        }

        assert(w.kind.has_compressor == (L >= 2));
        assert(w.kind.has_compressor == has(g, L, "attn_compressor_kv.weight"));
        assert((w.compressor_kv != nullptr) == w.kind.has_compressor);
        if (w.kind.has_compressor) {
            // CSA (even, ratio 4) projects to 2*head_dim; HCA (odd, ratio 128)
            // to head_dim. Both shape families are pinned here because a loader
            // that picked one for all 41 layers would fail on 20 or 21 of them.
            const uint64_t cd = (L % 2 == 0) ? 1024 : 512;
            assert(w.kind.compress_ratio == (L % 2 == 0 ? 4u : 128u));
            assert(w.kind.compressor_dim == cd);
            assert(w.compressor_kv->shape[0] == 4096 && w.compressor_kv->shape[1] == cd);
            assert(w.compressor_gate->shape[0] == 4096 && w.compressor_gate->shape[1] == cd);
            assert(w.compressor_ape->shape[0] == cd);
            assert(w.compressor_ape->shape[1] == w.kind.compress_ratio);
            assert(w.compressor_norm->shape[0] == 512);
            ++n_compressor;
        } else {
            assert(w.kind.compress_ratio == 0 && w.kind.compressor_dim == 0);
        }

        assert(w.kind.has_indexer == (L >= 2 && L % 2 == 0));
        assert(w.kind.has_indexer == has(g, L, "indexer.proj.weight"));
        assert((w.indexer_q_b != nullptr) == w.kind.has_indexer);
        if (w.kind.has_indexer) {
            assert(w.indexer_q_b->shape[0] == 1024 && w.indexer_q_b->shape[1] == 8192);
            assert(w.indexer_proj->shape[0] == 4096 && w.indexer_proj->shape[1] == 64);
            assert(w.indexer_compressor_kv->shape[1] == 256);
            assert(w.indexer_compressor_ape->shape[0] == 256);
            assert(w.indexer_compressor_ape->shape[1] == 4);
            assert(w.indexer_compressor_norm->shape[0] == 128);
            ++n_indexer;
        } else {
            assert(!w.indexer_proj && !w.indexer_compressor_kv &&
                   !w.indexer_compressor_gate && !w.indexer_compressor_ape &&
                   !w.indexer_compressor_norm);
        }
    }
    ok(std::string(label) + ": 3 hash / 41 compressor / 21 indexer layers",
       n_hash == 3 && n_compressor == 41 && n_indexer == 21);

    // -- EVERY bound tensor's byte size works out under ITS OWN dtype --------
    // This is the property a per-role dtype constant cannot have: bytes_for()
    // is evaluated with the tensor's own type_id, so if the loader had bound a
    // role-wide dtype the size would disagree for whichever tensor differs.
    {
        bool sizes_ok = true;
        std::string bad;
        for (const ie::GgufTensorInfo& ti : g.tensors()) {
            uint64_t hi = 1;
            for (uint32_t d = 1; d < ti.n_dims; ++d) hi *= ti.shape[d];
            const uint64_t want =
                uint64_t(ie::bytes_for(ti.dtype, size_t(ti.shape[0]))) * hi;
            if (want != ti.nbytes) {
                sizes_ok = false;
                bad = std::string(ti.name) + " (" + std::string(ie::type_name(ti.dtype)) + ")";
                break;
            }
        }
        ok(std::string(label) + ": every tensor's nbytes matches its OWN type_id", sizes_ok, bad);
    }

    // -- the always-resident set, from the BOUND POINTERS -------------------
    // Charged exactly as DeepSeek4Runtime::load charges it: upload_dense for the
    // 2-D dense weights, fp32 for the router, verbatim bytes for the F32/I32
    // vectors. Reading it off `m` rather than off the file's name table is the
    // point — these are the tensors load() actually selected.
    auto dense = [&](const ie::GgufTensorInfo* t) {
        assert(t && t->n_dims == 2);
        const uint64_t K = t->shape[0], N = t->shape[1];
        ok(std::string(label) + ": " + std::string(t->name) + " has an upload path",
           ie::ds4_dense_uploadable(t->dtype), tn(t));
        out.resident_device += ie::ds4_dense_device_bytes(t->dtype, K, N);
        out.resident_gguf   += t->nbytes;
        (ie::ds4_dense_keeps_packed(t->dtype, K) ? out.n_packed : out.n_expanded)++;
    };
    auto raw = [&](const ie::GgufTensorInfo* t) {
        assert(t);
        // f32vec/gvec memcpy these verbatim into a float/int32 array, so their
        // dtype is NOT negotiable and load() still binds them at exactly F32/I32.
        ok(std::string(label) + ": " + std::string(t->name) + " is still exactly F32/I32",
           t->dtype == ie::DType::kF32 || t->dtype == ie::DType::kI32, tn(t));
        out.resident_device += t->nbytes;
        out.resident_gguf   += t->nbytes;
    };
    dense(gl.token_embd); dense(gl.output);
    raw(gl.output_norm); raw(gl.output_hc_fn); raw(gl.output_hc_base); raw(gl.output_hc_scale);
    for (const auto& w : m.layers()) {
        dense(w.attn_q_a); dense(w.attn_q_b); dense(w.attn_kv);
        dense(w.attn_output_a); dense(w.attn_output_b);
        raw(w.attn_norm); raw(w.attn_q_a_norm); raw(w.attn_kv_a_norm); raw(w.attn_sinks);
        if (w.kind.has_compressor) {
            dense(w.compressor_kv); dense(w.compressor_gate);
            raw(w.compressor_ape); raw(w.compressor_norm);
        }
        if (w.kind.has_indexer) {
            dense(w.indexer_q_b); dense(w.indexer_proj);
            dense(w.indexer_compressor_kv); dense(w.indexer_compressor_gate);
            raw(w.indexer_compressor_ape); raw(w.indexer_compressor_norm);
        }
        raw(w.ffn_norm);
        // The router is deliberately widened to fp32 and stays that way.
        out.resident_device += uint64_t(cfg.n_experts) * cfg.hidden * 4;
        out.resident_gguf   += w.ffn_gate_inp->nbytes;
        dense(w.ffn_gate_shexp); dense(w.ffn_up_shexp); dense(w.ffn_down_shexp);
        if (w.kind.hash_router) raw(w.ffn_gate_tid2eid); else raw(w.exp_probs_b);
        raw(w.hc_attn_fn); raw(w.hc_attn_base); raw(w.hc_attn_scale);
        raw(w.hc_ffn_fn);  raw(w.hc_ffn_base);  raw(w.hc_ffn_scale);
    }

    // -- routed experts: slot layout from each tensor's own dtype -----------
    out.slot_bytes.assign(m.layers().size(), 0);
    for (size_t l = 0; l < m.layers().size(); ++l) {
        const auto& w = m.layers()[l];
        ie::Ds4SlotLayout lay;
        const std::string le = ie::ds4_slot_layout(*w.ffn_gate_exps, *w.ffn_up_exps,
                                                   *w.ffn_down_exps, cfg.hidden,
                                                   cfg.expert_ffn, lay);
        assert(le.empty());
        out.slot_bytes[l] = lay.bytes;
        out.per_layer    += lay.bytes;
    }
    out.pool = out.per_layer * cfg.n_experts;

    out.opened = true;
    return true;
}

}  // namespace

int main() {
    std::printf("deepseek4_model_test — one loader, both quants (per-tensor dtype binding)\n");

    Bound q3, q8;
    const bool present_q3 = check_model("UD-Q3_K_XL", env_or("DS4_GGUF", kQ3Shard1), q3);
    const bool present_q8 = check_model("UD-Q8_K_XL", env_or("DS4_GGUF_Q8", kQ8Shard1), q8);

    if (!present_q3 && !present_q8) {
        std::puts("deepseek4_model_test: SKIPPED (neither quant present)");
        return 0;
    }
    // Present-but-broken is a FAILURE, never a skip.
    const bool have_q3 = q3.opened, have_q8 = q8.opened;
    if (present_q3 && !have_q3) ok("UD-Q3_K_XL is present, so it must bind", false);
    if (present_q8 && !have_q8) ok("UD-Q8_K_XL is present, so it must bind", false);

    // ---------------------------------------------------------------------
    std::printf("\n[traps] the specific per-tensor dtypes a role constant gets wrong\n");
    // ---------------------------------------------------------------------
    if (have_q3) {
        ie::GgufReader g;
        assert(g.open(env_or("DS4_GGUF", kQ3Shard1)).empty());
        ie::DeepSeek4Config cfg;
        assert(ie::read_deepseek4_config(g, cfg).empty());
        ie::DeepSeek4Model m;
        assert(m.load(g, cfg).empty());

        ok("Q3: token_embd is Q8_0", m.globals().token_embd->dtype == ie::DType::kQ8_0,
           tn(m.globals().token_embd));
        ok("Q3: output is Q6_K", m.globals().output->dtype == ie::DType::kQ6_K,
           tn(m.globals().output));
        ok("Q3: blk.5 attn_q_b is Q8_0", m.layers()[5].attn_q_b->dtype == ie::DType::kQ8_0,
           tn(m.layers()[5].attn_q_b));
        ok("Q3: blk.4 indexer.proj is F32",
           m.layers()[4].indexer_proj->dtype == ie::DType::kF32, tn(m.layers()[4].indexer_proj));
        ok("Q3: blk.5 ffn_gate_shexp is Q8_0",
           m.layers()[5].ffn_gate_shexp->dtype == ie::DType::kQ8_0,
           tn(m.layers()[5].ffn_gate_shexp));

        // The Phase 1 expert property, re-asserted: blk.26 gate/up are MXFP4
        // while its neighbours are IQ3_XXS, and the byte size only works out
        // under each tensor's OWN dtype.
        std::set<uint32_t> mxfp4_gate;
        for (uint32_t L = 0; L < 43; ++L) {
            const auto& w = m.layers()[L];
            assert(w.ffn_down_exps->dtype == ie::DType::kMXFP4);
            assert(w.ffn_gate_exps->dtype == w.ffn_up_exps->dtype);
            if (w.ffn_gate_exps->dtype == ie::DType::kMXFP4) mxfp4_gate.insert(L);
            else assert(w.ffn_gate_exps->dtype == ie::DType::kIQ3_XXS);
        }
        ok("Q3: blk.26 gate/up are MXFP4 and it is the ONLY such layer",
           mxfp4_gate.size() == 1 && *mxfp4_gate.begin() == 26);
        ok("Q3: blk.25 and blk.27 gate/up are IQ3_XXS",
           m.layers()[25].ffn_gate_exps->dtype == ie::DType::kIQ3_XXS &&
           m.layers()[27].ffn_gate_exps->dtype == ie::DType::kIQ3_XXS,
           std::string(tn(m.layers()[25].ffn_gate_exps)) + " / " +
           tn(m.layers()[27].ffn_gate_exps));
        {   // The falsifier: assuming the sibling dtype for blk.26 gives a wrong size.
            const auto* t26 = m.layers()[26].ffn_gate_exps;
            const uint64_t hi = uint64_t(t26->shape[1]) * t26->shape[2];
            ok("Q3: blk.26's nbytes is WRONG under blk.25's dtype (a role constant would break)",
               t26->nbytes !=
                   uint64_t(ie::bytes_for(ie::DType::kIQ3_XXS, size_t(t26->shape[0]))) * hi);
        }

        // -- the loader still REFUSES a mis-stated config, through the new path -
        // bind_dense must not have become permissive about anything but dtype.
        auto refuses = [&](const char* what, ie::DeepSeek4Config bad) {
            ie::DeepSeek4Model m2;
            const std::string e = m2.load(g, bad);
            ok(std::string("Q3: REFUSED — ") + what, !e.empty(), e);
        };
        { auto b = cfg; b.expert_ffn = 4096;        refuses("expert_ffn 2048->4096", b); }
        { auto b = cfg; b.hash_layer_count = 2;     refuses("hash_layer_count 3->2", b); }
        // These four land on bind_dense tensors specifically.
        { auto b = cfg; b.vocab = 129279;           refuses("vocab off by one (token_embd/output)", b); }
        { auto b = cfg; b.q_lora_rank = 512;        refuses("q_lora_rank 1024->512 (attn_q_a/q_b)", b); }
        { auto b = cfg; b.n_layers = 44;            refuses("n_layers 43->44 (blk.43 not found)", b); }
        { auto b = cfg; b.compress_ratios[3] = 4;   refuses("compress_ratios[3] 128->4 (CSA/HCA flip)", b); }
    }

    if (have_q8) {
        ie::GgufReader g;
        assert(g.open(env_or("DS4_GGUF_Q8", kQ8Shard1)).empty());
        ie::DeepSeek4Config cfg;
        assert(ie::read_deepseek4_config(g, cfg).empty());
        ie::DeepSeek4Model m;
        const std::string e = m.load(g, cfg);
        ok("Q8: load() succeeds (THE BLOCKER)", e.empty(), e);
        if (e.empty()) {
            ok("Q8: attn_q_b is BF16, NOT Q8_0",
               m.layers()[5].attn_q_b->dtype == ie::DType::kBF16, tn(m.layers()[5].attn_q_b));
            ok("Q8: output is BF16, NOT Q6_K",
               m.globals().output->dtype == ie::DType::kBF16, tn(m.globals().output));
            ok("Q8: token_embd is BF16, NOT Q8_0",
               m.globals().token_embd->dtype == ie::DType::kBF16, tn(m.globals().token_embd));
            ok("Q8: indexer.proj is BF16, NOT F32 (the quietest of the four)",
               m.layers()[4].indexer_proj->dtype == ie::DType::kBF16,
               tn(m.layers()[4].indexer_proj));
            ok("Q8: ffn_down_shexp is BF16, NOT Q8_0",
               m.layers()[5].ffn_down_shexp->dtype == ie::DType::kBF16,
               tn(m.layers()[5].ffn_down_shexp));

            // Q8's routed experts are MXFP4 on ALL 43 layers — blk.26 is not
            // special here, which is the mirror image of Q3 and exactly why the
            // expert dispatch must be per tensor in BOTH directions.
            uint32_t mxfp4_layers = 0;
            for (uint32_t L = 0; L < 43; ++L) {
                const auto& w = m.layers()[L];
                if (w.ffn_gate_exps->dtype == ie::DType::kMXFP4 &&
                    w.ffn_up_exps->dtype   == ie::DType::kMXFP4 &&
                    w.ffn_down_exps->dtype == ie::DType::kMXFP4) ++mxfp4_layers;
            }
            ok("Q8: gate/up/down are MXFP4 on all 43 layers", mxfp4_layers == 43,
               std::to_string(mxfp4_layers) + " of 43");
            ok("Q8: blk.26 is NOT special — same slot size as blk.25, unlike Q3",
               q8.slot_bytes[26] == q8.slot_bytes[25],
               std::to_string(q8.slot_bytes[25]) + " vs " + std::to_string(q8.slot_bytes[26]));

            // The dense set really is all-BF16, so nothing qualifies for packing.
            ok("Q8: no dense weight stays packed (BF16 is already 2 B/element)",
               q8.n_packed == 0, std::to_string(q8.n_packed));
        }
    }
    if (have_q3)
        ok("Q3: 491 dense weights stay packed (490 Q8_0 + 1 Q6_K)", q3.n_packed == 491,
           std::to_string(q3.n_packed) + " packed / " + std::to_string(q3.n_expanded) + " expanded");
    if (have_q3 && have_q8)
        ok("both quants bind the SAME 1328 tensors through ONE code path",
           q3.n_bound == 1328 && q8.n_bound == 1328,
           std::to_string(q3.n_bound) + " / " + std::to_string(q8.n_bound));

    // ---------------------------------------------------------------------
    std::printf("\n[policy] ds4_dense_uploadable is a WHITELIST, not a wildcard\n");
    // ---------------------------------------------------------------------
    // The dtype gate is what keeps a mis-declared dtype a hard, named error
    // instead of an accept-anything path, so it is pinned exhaustively: exactly
    // the six dtypes upload_dense can consume, and nothing else.
    {
        const ie::DType accept[] = {ie::DType::kF32, ie::DType::kF16, ie::DType::kBF16,
                                    ie::DType::kQ8_0, ie::DType::kQ6_K, ie::DType::kQ4_K};
        bool all_accept = true;
        for (ie::DType d : accept) all_accept = all_accept && ie::ds4_dense_uploadable(d);
        ok("the six dtypes with an upload path are accepted", all_accept);

        uint32_t n_accept = 0;
        std::string accepted;
        for (uint32_t i = 0; i <= uint32_t(ie::DType::kCount); ++i) {
            const auto d = ie::DType(i);
            if (ie::ds4_dense_uploadable(d)) { ++n_accept; accepted += " " + std::string(ie::type_name(d)); }
        }
        ok("EXACTLY six dtypes are accepted over the whole enum", n_accept == 6,
           std::to_string(n_accept) + ":" + accepted);
        // The expert dtypes are the ones that must NOT slip into the dense path:
        // they have no host dequant here, so binding one would fail mid-upload.
        ok("MXFP4 is rejected (routed experts have their own path)",
           !ie::ds4_dense_uploadable(ie::DType::kMXFP4));
        ok("IQ3_XXS is rejected", !ie::ds4_dense_uploadable(ie::DType::kIQ3_XXS));
        ok("I32 is rejected (tid2eid is memcpy'd, never dequantised)",
           !ie::ds4_dense_uploadable(ie::DType::kI32));
        ok("the kCount sentinel is rejected", !ie::ds4_dense_uploadable(ie::DType::kCount));
    }

    // ---------------------------------------------------------------------
    std::printf("\n[residency] resident set and pinned host, from the REAL binding\n");
    // ---------------------------------------------------------------------
    for (const Bound* b : {&q3, &q8}) {
        if (!b->opened) continue;
        std::printf("    %s: resident device %s ; the file stores %s for that set\n"
                    "      (%llu dense weights packed, %llu expanded to fp16)\n"
                    "      routed experts: one slot per layer %llu B, pool %s\n",
                    b->label.c_str(), gb(b->resident_device).c_str(), gb(b->resident_gguf).c_str(),
                    (unsigned long long)b->n_packed, (unsigned long long)b->n_expanded,
                    (unsigned long long)b->per_layer, gb(b->pool).c_str());
        // The pinned-host figure the same way DeepSeek4Runtime::load derives it:
        // a 34.242 GB card, minus this resident set, minus the 2 GB workspace
        // margin, is the expert VRAM budget; ds4_plan_residency does the rest.
        constexpr uint64_t kB70GlobalMem = 34'242'000'000ull;
        constexpr uint64_t kWsMargin     = 2ull << 30;
        const uint64_t budget = kB70GlobalMem > b->resident_device + kWsMargin
                                    ? kB70GlobalMem - b->resident_device - kWsMargin : 0ull;
        ie::Ds4ResidencyPlan p;
        const std::string pe = ie::ds4_plan_residency(b->slot_bytes, 256, budget, 0,
                                                      b->pool, 12, p);
        std::printf("      expert VRAM budget %s -> %u slots/layer (%u static + %u stream),"
                    " pinned host %s\n",
                    gb(budget).c_str(), p.slots_per_layer, p.static_slots, p.stream_slots,
                    gb(p.host_bytes).c_str());
        ok(b->label + ": residency plan accepted at a cap of the whole pool", pe.empty(), pe);
        ok(b->label + ": static VRAM + pinned host == the whole expert pool (nothing double-counted)",
           uint64_t(p.static_slots) * b->per_layer + p.host_bytes == b->pool);
    }
    if (have_q3 && have_q8)
        ok("Q8's resident set is larger than Q3's (BF16 vs packed Q8_0/Q6_K)",
           q8.resident_device > q3.resident_device,
           gb(q8.resident_device) + " vs " + gb(q3.resident_device));

    std::printf("\ndeepseek4_model_test: %s (%d failures)\n", g_fail ? "FAIL" : "OK", g_fail);
    return g_fail ? 1 : 0;
}
