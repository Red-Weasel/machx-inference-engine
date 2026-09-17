// tests/unit/deepseek4_forward_test.cpp — Phase 5 gate: expert streaming +
// the assembled forward pass.
//
// WHAT THIS TEST CAN AND CANNOT PROVE, STATED UP FRONT
// ----------------------------------------------------
// §1-§3 run entirely on real hardware with real allocations and need NO model
// file: the pinned host arena, the per-card VRAM slot cache, the transfer
// queue, the measured H2D bandwidth and the measured DMA/compute overlap
// fraction.  Every number they print is measured in-process.
//
// §4 gates the two device primitives Phase 5 owns — `ds4_hc_mix` (the
// DecoderLayer residual assembly) and `ds4_hyper_head` (the final stream
// collapse) — against `$DS4_PARITY2_DIR`, i.e. against a real hooked
// DeepseekV4ForCausalLM forward.  It also proves the layer chaining and the
// `unsqueeze(2).expand(hc_mult)` stream initialisation directly from the dumped
// tensors.  These are the parts of the reference DecoderLayer / Model that no
// earlier phase covered.
//
// §5 attempts the REAL 128.20 GB model.  It reports what actually happened,
// including measured file-read throughput, and does NOT pretend a green exit
// means the model ran.
#undef NDEBUG
#include "ie/deepseek4.hpp"

#include "ie/deepseek4_experts.hpp"
#include "ie/engine.hpp"
#include "ie/expert_stream.hpp"
#include "ie/gguf.hpp"
#include "ie/gguf_writer.hpp"
#include "ie/quant_blocks.hpp"
#include "ie/model_config.hpp"
#include "ie/ops.hpp"

// The §6/§7 miniature fixture generator, shared verbatim with
// tools/ie_ds4_bench.cpp so the gate and the benchmark run the SAME model.
#include "../../tools/deepseek4/ds4_mini_gguf.hpp"

#include "nlohmann/json.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr const char* G = "\033[32m";
constexpr const char* R = "\033[31m";
constexpr const char* Y = "\033[33m";
constexpr const char* Z = "\033[0m";

int g_fail = 0;

bool ok(const char* name, bool cond, const std::string& detail = {}) {
    std::printf("  %s%-52s%s %s%s%s%s%s\n", cond ? G : R, name, Z,
                detail.empty() ? "" : "(", detail.c_str(), detail.empty() ? "" : ")  ",
                cond ? G : R, cond ? "OK" : "FAIL");
    std::printf("%s", Z);
    if (!cond) ++g_fail;
    return cond;
}

bool check_tol(const char* name, double observed, double bound) {
    const bool c = std::isfinite(observed) && observed <= bound;
    std::printf("  %s%-52s%s max|err|=%.3e  bound=%.3e  %s%s%s\n",
                c ? G : R, name, Z, observed, bound, c ? G : R, c ? "OK" : "FAIL", Z);
    if (!c) ++g_fail;
    return c;
}

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Host memory accounting taken from the kernel, so criterion 1's "actual host
// usage" is the OS's number and not the process's own bookkeeping.  Fields are
// kB in /proc, reported here in GB.
struct HostMem { double rss_gb = 0, locked_gb = 0, avail_gb = 0; };
HostMem host_mem() {
    HostMem h;
    auto scan = [](const char* path, const char* key, double& out_gb) {
        std::FILE* f = std::fopen(path, "r");
        if (!f) return;
        char line[256];
        const size_t kl = std::strlen(key);
        while (std::fgets(line, sizeof line, f))
            if (std::strncmp(line, key, kl) == 0) { out_gb = std::atof(line + kl) / 1e6; break; }
        std::fclose(f);
    };
    scan("/proc/self/status", "VmRSS:", h.rss_gb);
    scan("/proc/self/status", "VmLck:", h.locked_gb);
    scan("/proc/meminfo",     "MemAvailable:", h.avail_gb);
    return h;
}

sycl::queue pick_queue(uint32_t ordinal) {
    std::vector<sycl::device> gpus;
    for (const auto& d : sycl::device::get_devices())
        if (d.is_gpu() && d.get_backend() == sycl::backend::ext_oneapi_level_zero) gpus.push_back(d);
    if (gpus.empty()) { std::fprintf(stderr, "no Level Zero GPU\n"); std::exit(1); }
    const uint32_t o = std::min<uint32_t>(ordinal, uint32_t(gpus.size()) - 1);
    return sycl::queue(gpus[o], sycl::property_list{sycl::property::queue::in_order(),
                                                    sycl::property::queue::enable_profiling()});
}

// ---------------------------------------------------------------------------
// $DS4_TP_GPUS — the card->GPU map for a REAL two-card load
// ---------------------------------------------------------------------------
// "0,1" drives `DeepSeek4TpRuntime` over BOTH B70s: each card holds
// `moe_intermediate_size / n_cards` of EVERY routed expert plus a full mirrored
// copy of the non-expert set, so the VRAM of BOTH cards is filled (via
// `ds4_plan_residency_tp`, which halves the per-card expert bytes and divides
// the whole-box pinned-host cap by n_cards) before a byte spills to host RAM.
//
// Unset, or a single ordinal, is the single-card path this file has always run —
// unchanged, byte for byte.  A REPEATED ordinal is refused here rather than
// quietly enabling `Ds4TpOptions::same_device_rehearsal`: under rehearsal the
// residency plan is a fiction (each card sizes its arena from the whole device),
// and the entire point of this switch is to prove two cards really fill.
std::vector<uint32_t> parse_tp_gpus() {
    std::vector<uint32_t> v;
    const char* s = std::getenv("DS4_TP_GPUS");
    if (!s) return v;
    std::string cur;
    for (const char* p = s;; ++p) {
        if (*p == ',' || *p == 0) {
            if (!cur.empty()) v.push_back(uint32_t(std::atoi(cur.c_str())));
            cur.clear();
            if (*p == 0) break;
        } else cur.push_back(*p);
    }
    return v;
}

// Builds the orchestrator options from a single-card `Ds4Options`.  The three
// per-card fields (`device_ordinal`, `n_cards`, `card`) are owned by the
// orchestrator and MUST be left at their defaults, or `load()` refuses — so this
// clears them explicitly instead of hoping the caller did.
ie::Ds4TpOptions tp_options(const ie::Ds4Options& base, const std::vector<uint32_t>& gpus) {
    ie::Ds4TpOptions to;
    to.base                 = base;
    to.base.device_ordinal  = 0;
    to.base.n_cards         = 1;
    to.base.card            = 0;
    to.n_cards              = uint32_t(gpus.size());
    to.device_ordinals      = gpus;
    to.same_device_rehearsal = false;   // never silently; see parse_tp_gpus
    return to;
}

// Per-card residency report — the thing the owner needs to SEE: BOTH cards
// filling, and exactly what spilled to host.  Every figure is read back off the
// loaded runtimes (`residency()`, `arena()`, `cache()`), never modelled.
// Returns true iff every card actually holds device bytes.
bool report_cards(ie::DeepSeek4TpRuntime& tp, uint32_t n_experts, uint32_t expert_ffn) {
    uint64_t host_total = 0, vram_total = 0;
    bool all_nonzero = true;
    std::printf("    %s---- PER-CARD RESIDENCY (both cards must be non-zero) ----%s\n", G, Z);
    for (uint32_t c = 0; c < tp.n_cards(); ++c) {
        ie::DeepSeek4Runtime&       rt = tp.card(c);
        const ie::Ds4ResidencyPlan& p  = rt.residency();
        const ie::Ds4ExpertSlice&   s  = rt.expert_slice();
        const uint64_t static_vram = uint64_t(p.static_slots) * p.layer_slot_total;
        const uint64_t stream_vram = uint64_t(p.stream_slots) * p.layer_slot_total;
        const uint64_t card_vram   = rt.resident_bytes() + rt.cache().device_bytes();
        std::printf("    card %u -> GPU %u : %s\n", c, tp.device_of(c),
                    rt.queue().get_device().get_info<sycl::info::device::name>().c_str());
        std::printf("      resident set uploaded : %12.3f MB\n",
                    double(rt.resident_bytes()) / 1e6);
        std::printf("      static expert VRAM    : %12.3f MB   (%u slots/layer, never evicted,"
                    " no host copy)\n", double(static_vram) / 1e6, p.static_slots);
        std::printf("      streaming expert VRAM : %12.3f MB   (%u slots/layer)\n",
                    double(stream_vram) / 1e6, p.stream_slots);
        std::printf("      slots/layer           : %12u      (%.1f%% of %u experts; cache reports %u)\n",
                    p.slots_per_layer, 100.0 * p.slots_per_layer / std::max(n_experts, 1u),
                    n_experts, rt.cache().slots_per_layer());
        std::printf("      pinned host arena     : %12.3f MB   in %zu segments"
                    " (%u of %u experts/layer spilled)\n",
                    double(rt.arena().total_bytes()) / 1e6, rt.arena().n_segments(),
                    p.pinned_experts, n_experts);
        std::printf("      expert slice          : intermediate [%u,%u) of %u  (card %u of %u)\n",
                    s.ef0, s.ef0 + s.efc, expert_ffn, s.card, s.n_cards);
        std::printf("      CARD VRAM TOTAL       : %12.3f MB   (resident + expert arena)\n",
                    double(card_vram) / 1e6);
        if (card_vram == 0) all_nonzero = false;
        host_total += rt.arena().total_bytes();
        vram_total += card_vram;
    }
    std::printf("    %sBOX TOTAL: %.3f GB VRAM over %u cards | %.3f GB pinned host"
                " + %.3f MB cross-card staging%s\n", G,
                double(vram_total) / 1e9, tp.n_cards(), double(host_total) / 1e9,
                double(tp.stage_bytes()) / 1e6, Z);
    return all_nonzero;
}

// One handle over "single card" and "n cards in lockstep" so the §5 body below
// is ONE code path.  It dispatches; it computes nothing, so neither path can
// acquire behaviour the other does not have.
struct FwdDriver {
    ie::DeepSeek4Runtime*   single = nullptr;
    ie::DeepSeek4TpRuntime* tp     = nullptr;
    std::string forward(const int32_t* ids, uint32_t T, uint32_t pos0, float* lg) {
        return tp ? tp->forward(ids, T, pos0, lg) : single->forward(ids, T, pos0, lg);
    }
    void reset_context() { if (tp) tp->reset_context(); else single->reset_context(); }
    // Card 0 is the one whose logits and timings are the run's; under TP the
    // other cards' figures are printed separately by report_cards().
    ie::DeepSeek4Runtime& card0() { return tp ? tp->card(0) : *single; }
    uint32_t n_cards() const { return tp ? tp->n_cards() : 1u; }
};

// A GgufTensorInfo over a caller-owned buffer — lets §1-§3 exercise the real
// slot layout and the real packer without a 128 GB file.
struct FakeTensor {
    std::string             name;
    std::vector<uint8_t>    bytes;
    ie::GgufTensorInfo      ti{};
    FakeTensor(std::string nm, ie::DType dt, uint64_t K, uint64_t N, uint64_t E, uint64_t seed) {
        name = std::move(nm);
        ti.name = name;
        ti.dtype = dt; ti.n_dims = 3;
        ti.shape[0] = K; ti.shape[1] = N; ti.shape[2] = E;
        ti.nbytes = ie::bytes_for(dt, size_t(K)) * N * E;
        bytes.resize(size_t(ti.nbytes));
        std::mt19937_64 rng(seed);
        for (auto& b : bytes) b = uint8_t(rng() & 0xFF);
        ti.data = bytes.data();
    }
};

// ---------------------------------------------------------------------------
// parity blob plumbing (same conventions as deepseek4_parity_test.cpp)
// ---------------------------------------------------------------------------
using json = nlohmann::json;

std::string dir_or(const char* env, const char* fallback) {
    if (const char* v = std::getenv(env)) return std::string(v);
    return std::string(fallback);
}

struct Blob { std::string file; std::vector<size_t> shape; size_t numel = 0; };
struct Component {
    std::string name, cls;
    std::map<std::string, Blob> params, inputs, outputs;
};

Blob parse_blob(const json& j) {
    Blob b;
    b.file = j.at("file").get<std::string>();
    b.numel = j.at("numel").get<size_t>();
    for (const auto& d : j.at("shape")) b.shape.push_back(d.get<size_t>());
    return b;
}

std::vector<float> load_f32(const std::string& path, size_t expect) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    const size_t bytes = size_t(f.tellg());
    if (bytes != expect * 4) {
        std::fprintf(stderr, "%s: %zu bytes, expected %zu\n", path.c_str(), bytes, expect * 4);
        std::exit(1);
    }
    std::vector<float> v(expect);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), std::streamsize(bytes));
    return v;
}

template <typename T> T* to_dev(sycl::queue& q, const std::vector<T>& h) {
    T* p = sycl::malloc_device<T>(h.size(), q);
    q.memcpy(p, h.data(), h.size() * sizeof(T)).wait();
    return p;
}
template <typename T> std::vector<T> from_dev(sycl::queue& q, const T* p, size_t n) {
    std::vector<T> h(n);
    q.memcpy(h.data(), p, n * sizeof(T)).wait();
    return h;
}
double max_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, double(std::fabs(a[i] - b[i])));
    return m;
}
double max_abs(const std::vector<float>& a) {
    double m = 0;
    for (float x : a) m = std::max(m, double(std::fabs(x)));
    return m;
}


// ---------------------------------------------------------------------------
// §6 helpers — a miniature but STRUCTURALLY REAL deepseek4 GGUF.
//
// The generator MOVED to tools/deepseek4/ds4_mini_gguf.hpp, unchanged, because
// tools/ie_ds4_bench.cpp needs the SAME fixture and two copies of a 230-line
// weight generator is two things that can drift apart.  `namespace mini` is
// retained as an alias so every §6/§7 call site below reads exactly as before.
// Read that header for why the fixture exists and what it can and cannot prove.
namespace mini = ds4mini;

// ---------------------------------------------------------------------------
// §8 helpers — give the miniature fixture a WORKING tokenizer.
// ---------------------------------------------------------------------------
// §8 drives the fixture through ie::Engine, which needs the tokenizer the
// fixture does not have: ds4_mini_gguf.hpp writes `tokenizer.ggml.tokens` =
// {"t0"…"t511"} and NO `tokenizer.ggml.merges`, so Tokenizer::load_from_gguf
// refuses ("merges missing"), and even if it did not, that vocab holds no
// single-character token, so GPT-2 BPE can encode NOTHING (bpe_merge_word emits
// -1 for every symbol and encode() returns an empty id list).
//
// The fixture header is owned by §6/§7 and by tools/ie_ds4_bench.cpp, and its
// output is frozen (see its own comment), so this does NOT change it.  It
// rewrites a COPY of the written file, inserting three KV pairs at the FRONT of
// the KV block:
//   * tokenizer.ggml.tokens — 512 entries, which SHADOWS the fixture's own
//     (GgufReader::find_kv returns the FIRST match).  512 exactly, because
//     read_deepseek4_config takes cfg.vocab from this array's length and the
//     fixture's token_embd/output are [H, 512].
//       ids   0..94  = the GPT-2 byte-encoding of bytes 32..126, so every
//                      printable-ASCII prompt encodes, and every one of those
//                      ids decodes back to its byte.
//       ids  95..511 = two printable-ASCII letters, so a sampled id from the
//                      other 81% of the vocab still decodes to readable text
//                      rather than to a broken UTF-8 fragment.
//   * tokenizer.ggml.merges — empty.  Present because load_from_gguf REQUIRES
//     the key for a BPE vocab; empty because with no merges every character is
//     its own token, which is what makes the encode/decode round trip exact.
//   * a pad string, whose only job is to make the inserted block a whole number
//     of alignment units (see below).
//
// WHY THAT IS SAFE.  Tensor-info offsets are relative to the start of the data
// blob, and the reader puts that blob at align_up(end_of_tensor_infos).  The
// original blob therefore already starts at a multiple of the alignment, so if
// the inserted bytes are ALSO a multiple of it, every tensor lands at exactly
// its old offset and the rest of the file is copied verbatim.  Nothing else in
// the header moves relative to anything it points at.
void kv_put_u64(std::vector<uint8_t>& b, uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back(uint8_t(v >> (8 * i)));
}
void kv_put_u32(std::vector<uint8_t>& b, uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(uint8_t(v >> (8 * i)));
}
void kv_put_str(std::vector<uint8_t>& b, const std::string& s) {
    kv_put_u64(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}
void kv_string(std::vector<uint8_t>& b, const std::string& key, const std::string& val) {
    kv_put_str(b, key);
    kv_put_u32(b, uint32_t(ie::GgufValueType::kString));
    kv_put_str(b, val);
}
void kv_string_array(std::vector<uint8_t>& b, const std::string& key,
                     const std::vector<std::string>& vals) {
    kv_put_str(b, key);
    kv_put_u32(b, uint32_t(ie::GgufValueType::kArray));
    kv_put_u32(b, uint32_t(ie::GgufValueType::kString));
    kv_put_u64(b, vals.size());
    for (const auto& v : vals) kv_put_str(b, v);
}

// The 512 tokens described above.  Byte 32 (space) is the only one of 32..126
// that GPT-2 does not map to itself — it becomes U+0120 'Ġ' — because bytes
// 33..126 are the identity range of the byte encoder (tokenizer.cpp
// build_byte_maps) and everything outside it is remapped to 256+n in ascending
// byte order, which puts 32 at codepoint 288.
std::vector<std::string> mini_engine_vocab() {
    std::vector<std::string> v;
    v.reserve(512);
    v.push_back("\xc4\xa0");                                   // byte 32 -> U+0120
    for (int b = 33; b <= 126; ++b) v.push_back(std::string(1, char(b)));   // 94 -> ids 1..94
    for (int j = 0; v.size() < 512; ++j)                                    // 417 fillers
        v.push_back(std::string{char('A' + (j % 26)), char('a' + (j / 26))});
    return v;
}

// Copy `src` to `dst`, inserting the three KVs.  Returns "" or a diagnostic.
std::string write_tokenizer_patched_gguf(const std::string& src, const std::string& dst) {
    uint64_t align = 32;
    {
        ie::GgufReader g;
        if (auto e = g.open(src); !e.empty()) return "reopen fixture: " + e;
        align = g.alignment();
    }
    std::ifstream in(src, std::ios::binary);
    if (!in) return "open " + src;
    std::vector<uint8_t> raw((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
    if (raw.size() < 24) return "fixture shorter than a GGUF header";

    std::vector<uint8_t> ins;
    kv_string_array(ins, "tokenizer.ggml.tokens", mini_engine_vocab());
    kv_string_array(ins, "tokenizer.ggml.merges", {});
    // Pad KV: sized so (ins.size() + this KV) is a whole number of alignment
    // units.  Its own body is 8+key+4+8+n bytes, so solve for n.
    const std::string pad_key = "general.ie_engine_fixture_pad";
    const size_t fixed = ins.size() + 8 + pad_key.size() + 4 + 8;
    const size_t n_pad = (align - (fixed % align)) % align;
    kv_string(ins, pad_key, std::string(n_pad, 'x'));
    if (ins.size() % align != 0) return "pad arithmetic wrong";

    uint64_t n_kv = 0;
    std::memcpy(&n_kv, raw.data() + 16, 8);
    n_kv += 3;
    std::vector<uint8_t> out;
    out.reserve(raw.size() + ins.size());
    out.insert(out.end(), raw.begin(), raw.begin() + 16);
    kv_put_u64(out, n_kv);
    out.insert(out.end(), ins.begin(), ins.end());
    out.insert(out.end(), raw.begin() + 24, raw.end());

    std::ofstream o(dst, std::ios::binary);
    if (!o) return "open " + dst;
    o.write(reinterpret_cast<const char*>(out.data()), std::streamsize(out.size()));
    o.close();
    return o ? std::string{} : std::string("write " + dst);
}

// Printable form of a generated string: the fixture's weights are random, so the
// text is meant to be judged as "did the tokenizer round-trip", not as language.
// Escaping keeps a stray control byte from corrupting the terminal.
std::string printable(std::string_view s) {
    std::string o;
    for (unsigned char c : s) {
        if (c == '\\') o += "\\\\";
        else if (c >= 0x20 && c < 0x7f) o.push_back(char(c));
        else { char b[8]; std::snprintf(b, sizeof(b), "\\x%02x", c); o += b; }
    }
    return o;
}

}  // namespace

int main() {
    using namespace ie;

    // Real V4-Flash expert geometry (docs/deepseek4/21_tensor_manifest_verified.md).
    constexpr uint32_t H = 4096, EF = 2048, NEXP = 256;

    std::printf("\n=== DeepSeek-V4 Phase 5 gate: expert streaming + forward assembly ===\n");

    sycl::queue q = pick_queue(std::getenv("DS4_GPU") ? uint32_t(std::atoi(std::getenv("DS4_GPU"))) : 1);
    std::printf("device        : %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const uint64_t maxalloc = q.get_device().get_info<sycl::info::device::max_mem_alloc_size>();
    const uint64_t gmem     = q.get_device().get_info<sycl::info::device::global_mem_size>();
    std::printf("global_mem    : %.2f GB     max_mem_alloc_size: %.2f GB\n",
                double(gmem) / 1e9, double(maxalloc) / 1e9);

    // The card->GPU map, decided ONCE and announced, so no run can be mistaken
    // for the other.  §5 (the real model) and §7 (the miniature fixture) both
    // read it; §1-§4 and §6 are single-card and unaffected.
    const std::vector<uint32_t> tp_gpus = parse_tp_gpus();
    bool tp_mode = tp_gpus.size() > 1;
    if (tp_mode) {
        for (size_t a = 0; a < tp_gpus.size() && tp_mode; ++a)
            for (size_t b = a + 1; b < tp_gpus.size(); ++b)
                if (tp_gpus[a] == tp_gpus[b]) {
                    std::printf("  %sDS4_TP_GPUS names GPU %u twice (cards %zu and %zu).  REFUSED: each"
                                " card sizes its expert arena from that device's whole"
                                " global_mem_size, so two cards on one device would print a residency"
                                " plan neither can honour.  Same-device rehearsal lives in"
                                " deepseek4_residency_test (Ds4TpOptions::same_device_rehearsal); it"
                                " is deliberately not reachable from this gate, whose entire purpose"
                                " is to prove BOTH cards fill.%s\n", R, tp_gpus[a], a, b, Z);
                    ++g_fail;
                    tp_mode = false;
                    break;
                }
    }
    if (tp_mode) {
        std::printf("%sDS4_TP_GPUS   : card->GPU map {", Y);
        for (size_t c = 0; c < tp_gpus.size(); ++c)
            std::printf("%s%u", c ? ", " : "", tp_gpus[c]);
        std::printf("} — %zu-CARD TENSOR-PARALLEL RUN (§5 real model, §7 miniature)%s\n",
                    tp_gpus.size(), Z);
    } else if (tp_gpus.size() > 1) {
        std::printf("%sDS4_TP_GPUS   : REFUSED (see above).  §5 and §7 will NOT run; no two-card"
                    " result is reported.%s\n", R, Z);
    } else if (tp_gpus.size() == 1) {
        std::printf("DS4_TP_GPUS   : one ordinal (%u) — SINGLE CARD, identical to the default path\n",
                    tp_gpus[0]);
    } else {
        std::printf("DS4_TP_GPUS   : unset — single card.  Set DS4_TP_GPUS=0,1 to fill BOTH GPUs.\n");
    }

    // -----------------------------------------------------------------------
    std::printf("\n[1] Slot layout — memcpy-deliverable, per-tensor dtype\n");
    // -----------------------------------------------------------------------
    // blk.25-shaped (IQ3_XXS gate/up + MXFP4 down) and blk.26-shaped (MXFP4
    // everywhere).  Nothing below branches on the layer index: the layout comes
    // out of each tensor's own dtype.
    FakeTensor g25("blk.25.ffn_gate_exps.weight", DType::kIQ3_XXS, H, EF, 2, 1);
    FakeTensor u25("blk.25.ffn_up_exps.weight",   DType::kIQ3_XXS, H, EF, 2, 2);
    FakeTensor d25("blk.25.ffn_down_exps.weight", DType::kMXFP4,   EF, H, 2, 3);
    FakeTensor g26("blk.26.ffn_gate_exps.weight", DType::kMXFP4,   H, EF, 2, 4);
    FakeTensor u26("blk.26.ffn_up_exps.weight",   DType::kMXFP4,   H, EF, 2, 5);
    FakeTensor d26("blk.26.ffn_down_exps.weight", DType::kMXFP4,   EF, H, 2, 6);

    Ds4SlotLayout lay25{}, lay26{};
    ok("slot layout, mixed IQ3_XXS + MXFP4 layer",
       ds4_slot_layout(g25.ti, u25.ti, d25.ti, H, EF, lay25).empty());
    ok("slot layout, all-MXFP4 layer (blk.26 shape)",
       ds4_slot_layout(g26.ti, u26.ti, d26.ti, H, EF, lay26).empty());
    std::printf("    normal-layer slot = %llu B (%.3f MB)   blk.26 slot = %llu B (%.3f MB)\n",
                (unsigned long long)lay25.bytes, double(lay25.bytes) / 1e6,
                (unsigned long long)lay26.bytes, double(lay26.bytes) / 1e6);
    // 42 normal layers x 256 experts + 1 wide layer x 256 experts.
    const uint64_t pool_total = (42ull * lay25.bytes + lay26.bytes) * NEXP;
    std::printf("    full V4-Flash expert pool = %.3f GB\n", double(pool_total) / 1e9);
    ok("blk.26 slot is wider than a normal layer's (UD mixed precision)",
       lay26.bytes > lay25.bytes);
    ok("pool within 2% of the documented 120.393 GB",
       std::fabs(double(pool_total) - 120.393e9) / 120.393e9 < 0.02,
       std::to_string(double(pool_total) / 1e9) + " GB");

    // Packed bytes must reproduce exactly what ds4_expert_bank_upload would
    // have produced — the fetch path is a memcpy, so if the packing differs the
    // GEMV reads garbage.
    {
        std::vector<uint8_t> slot(lay25.bytes);
        ok("ds4_slot_pack (IQ3+IQ3+MXFP4)",
           ds4_slot_pack(lay25, g25.ti, u25.ti, d25.ti, 1, slot.data()).empty());
        DS4ExpertBank ref{};
        const std::string e = ds4_expert_bank_upload(q, g25.ti, H, EF, 2, ref);
        ok("reference bank upload", e.empty(), e);
        // Compare expert 1's gp/ap/dp planes against the packed slot.
        const DS4ExpertBank view = ds4_slot_bank(lay25.gate, slot.data());
        auto gh = from_dev(q, ref.gp + ref.gp_stride, size_t(ref.gp_stride));
        auto ah = from_dev(q, ref.ap + ref.ap_stride, size_t(ref.ap_stride));
        auto dh = from_dev(q, ref.dp + ref.dp_stride, size_t(ref.dp_stride));
        ok("packed gp plane == ds4_expert_bank_upload's",
           std::memcmp(gh.data(), view.gp, gh.size()) == 0);
        ok("packed ap plane == ds4_expert_bank_upload's",
           std::memcmp(ah.data(), view.ap, ah.size() * 4) == 0);
        ok("packed dp plane == ds4_expert_bank_upload's",
           std::memcmp(dh.data(), view.dp, dh.size() * 2) == 0);
        ds4_expert_bank_free(q, ref);
    }

    // -----------------------------------------------------------------------
    std::printf("\n[2] Segmented pinned host arena (criterion 5)\n");
    // -----------------------------------------------------------------------
    ok("kDs4MaxAllocBytes matches the device's reported ceiling within 1%",
       std::fabs(double(kDs4MaxAllocBytes) - double(maxalloc)) / double(maxalloc) < 0.01,
       std::to_string(double(maxalloc) / 1e9) + " GB reported");

    // A segment target ABOVE the ceiling must be refused, not clamped.
    {
        Ds4HostArena bad;
        const std::string e = bad.init(q, std::vector<uint64_t>{lay25.bytes}, NEXP,
                                       kDs4MaxAllocBytes + (1ull << 30));
        ok("segment target above the 32.53 GB cap is REFUSED", !e.empty());
        std::printf("    diagnostic: %s\n", e.c_str());
    }
    // A single layer that alone exceeds the ceiling must be refused by name.
    {
        Ds4HostArena bad;
        const std::string e = bad.init(q, std::vector<uint64_t>{kDs4MaxAllocBytes / 4}, 8, 1ull << 30);
        ok("a layer needing > 32.53 GB in one piece is REFUSED", !e.empty());
        std::printf("    diagnostic: %s\n", e.c_str());
    }

    // A real arena.  The full 120.4 GB pool is not allocated here (the test must
    // stay runnable alongside a desktop session); a scaled subset with the REAL
    // slot sizes exercises the identical code path.
    const uint32_t arena_layers  = 8;
    const uint32_t arena_experts = 64;
    std::vector<uint64_t> slot_bytes(arena_layers, lay25.bytes);
    slot_bytes[arena_layers - 1] = lay26.bytes;         // one wide layer, like blk.26
    Ds4HostArena arena;
    {
        const double t0 = now_s();
        const std::string e = arena.init(q, slot_bytes, arena_experts, 2ull << 30);
        const double dt = now_s() - t0;
        ok("pinned arena init", e.empty(), e);
        std::printf("    %.3f GB pinned in %zu segments, largest %.3f GB, %.2f s (%.2f GB/s)\n",
                    double(arena.total_bytes()) / 1e9, arena.n_segments(),
                    double(arena.max_segment_bytes()) / 1e9, dt,
                    double(arena.total_bytes()) / 1e9 / std::max(dt, 1e-9));
        bool under = true;
        for (size_t i = 0; i < arena.n_segments(); ++i)
            under = under && arena.segment_bytes(i) < kDs4MaxAllocBytes;
        ok("every segment is below the per-allocation ceiling", under);
        // No expert straddles a segment: slot(L,e) + slot_bytes must stay inside
        // the same segment, which whole-layer packing guarantees.  Check it by
        // proving consecutive experts are exactly slot_bytes apart.
        bool contiguous = true;
        for (uint32_t l = 0; l < arena_layers; ++l)
            for (uint32_t e2 = 1; e2 < arena_experts; ++e2) {
                const auto a = reinterpret_cast<uintptr_t>(arena.slot(l, e2 - 1));
                const auto b = reinterpret_cast<uintptr_t>(arena.slot(l, e2));
                contiguous = contiguous && (b - a == slot_bytes[l]);
            }
        ok("experts are contiguous within a layer (no straddling)", contiguous);
    }

    // Fill it with the real packer so the bytes moved later are real weights.
    for (uint32_t l = 0; l < arena_layers; ++l) {
        const bool wide = (l == arena_layers - 1);
        for (uint32_t e2 = 0; e2 < arena_experts; ++e2) {
            const std::string s = wide
                ? ds4_slot_pack(lay26, g26.ti, u26.ti, d26.ti, e2 % 2, arena.slot(l, e2))
                : ds4_slot_pack(lay25, g25.ti, u25.ti, d25.ti, e2 % 2, arena.slot(l, e2));
            if (!s.empty()) { ok("arena fill", false, s); break; }
        }
    }
    // Tag each slot so a fetch can be proven to have moved the RIGHT expert.
    for (uint32_t l = 0; l < arena_layers; ++l)
        for (uint32_t e2 = 0; e2 < arena_experts; ++e2)
            static_cast<uint32_t*>(arena.slot(l, e2))[0] = l * 1000u + e2;

    // -----------------------------------------------------------------------
    std::printf("\n[3] Per-card VRAM slot cache, transfer queue, measured DMA\n");
    // -----------------------------------------------------------------------
    Ds4ExpertCache cache;
    const uint32_t slots = 16;
    {
        const std::string e = cache.init(q, arena, slots);
        ok("cache init", e.empty(), e);
        std::printf("    %u slots/layer x %u layers = %.3f GB VRAM (%.1f%% residency of %u experts)\n",
                    cache.slots_per_layer(), arena_layers, double(cache.device_bytes()) / 1e9,
                    100.0 * cache.slots_per_layer() / arena_experts, arena_experts);
        ok("transfer queue is a DISTINCT in-order queue on the SAME context",
           cache.transfer_queue().get_context() == q.get_context() &&
           cache.transfer_queue().is_in_order() &&
           cache.transfer_queue() != q);
    }

    // Correctness: after acquire, the slot bytes must equal the arena bytes.
    {
        const int32_t ids[6] = {3, 11, 40, 7, 3, 62};
        uint32_t sl[6];
        cache.acquire(0, ids, 6, sl).wait();
        bool tags_ok = true, bytes_ok = true;
        std::vector<uint8_t> got(slot_bytes[0]);
        for (uint32_t k = 0; k < 6; ++k) {
            tags_ok = tags_ok && cache.tag(0, sl[k]) == ids[k];
            q.memcpy(got.data(), cache.slot_ptr(0, sl[k]), slot_bytes[0]).wait();
            bytes_ok = bytes_ok &&
                       std::memcmp(got.data(), arena.slot(0, uint32_t(ids[k])), slot_bytes[0]) == 0;
        }
        ok("directory tags match the requested expert ids", tags_ok);
        ok("fetched device bytes == pinned host bytes (byte-exact)", bytes_ok);
        ok("a repeated id resolves to the SAME slot (no double fetch)", sl[0] == sl[4]);
        const auto& s = cache.stats();
        ok("6 ids, 5 distinct -> 5 misses + 1 hit", s.misses == 5 && s.hits == 1,
           std::to_string(s.misses) + " miss / " + std::to_string(s.hits) + " hit");
    }

    // Measured H2D bandwidth on the transfer queue at the real per-expert size.
    {
        cache.reset_stats();
        const uint32_t rounds = 8;
        std::vector<int32_t> ids(slots);
        double t0 = now_s();
        for (uint32_t r = 0; r < rounds; ++r) {
            for (uint32_t i = 0; i < slots; ++i) ids[i] = int32_t((r * slots + i) % arena_experts);
            uint32_t sl[64];
            cache.acquire(1, ids.data(), slots, sl).wait();
        }
        const double dt = now_s() - t0;
        cache.collect_dma_time();
        const auto& s = cache.stats();
        std::printf("    fetched %.3f GB in %u transfers: wall %.1f ms -> %.2f GB/s ; "
                    "transfer-queue busy %.1f ms -> %.2f GB/s\n",
                    double(s.bytes_fetched) / 1e9, unsigned(s.misses), dt * 1e3,
                    double(s.bytes_fetched) / 1e9 / dt, s.dma_seconds * 1e3,
                    double(s.bytes_fetched) / 1e9 / std::max(s.dma_seconds, 1e-9));
        ok("pinned H2D > 15 GB/s (design assumes ~25 GB/s)",
           double(s.bytes_fetched) / 1e9 / std::max(s.dma_seconds, 1e-9) > 15.0);
    }

    // MEASURED DMA/compute overlap fraction, with the launch-heavy compute shape
    // decode actually has.
    {
        const uint32_t K = 4096, N = 2048;
        auto* wf16 = sycl::malloc_device<sycl::half>(size_t(K) * N, q);
        auto* xf32 = sycl::malloc_device<float>(K, q);
        auto* yf32 = sycl::malloc_device<float>(N, q);
        q.memset(wf16, 0x11, size_t(K) * N * 2);
        q.memset(xf32, 0, K * 4).wait();
        // Sized so the compute side is COMPARABLE to the copy side.  If the
        // kernel is far shorter than the copy the overlap ratio is dominated by
        // timer noise and can even exceed 1.0, which says nothing.
        uint32_t nk = 24;

        auto compute = [&] {
            for (uint32_t i = 0; i < nk; ++i) {
                DS4ExpertBank b{};   // unused; keep the shape honest with a real GEMV-ish kernel
                (void)b;
                q.submit([&](sycl::handler& h) {
                    h.parallel_for(sycl::nd_range<1>(size_t(N) * 128, 128), [=](sycl::nd_item<1> it) {
                        const uint32_t n = uint32_t(it.get_group(0));
                        const uint32_t l = uint32_t(it.get_local_id(0));
                        float acc = 0.f;
                        for (uint32_t k = l; k < K; k += 128)
                            acc += float(wf16[uint64_t(n) * K + k]) * xf32[k];
                        acc = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
                        if (l == 0) yf32[n] = acc;
                    });
                });
            }
        };
        auto copy_only = [&] {
            std::vector<int32_t> ids(slots);
            for (uint32_t i = 0; i < slots; ++i) ids[i] = int32_t(i + 17) % int32_t(arena_experts);
            uint32_t sl[64];
            return cache.acquire(2, ids.data(), slots, sl);
        };

        compute(); q.wait();                                  // warm up
        double c0 = now_s(); compute(); q.wait();
        double t_kernel = now_s() - c0;
        // One calibration step: scale the launch count so t_kernel lands in the
        // same order of magnitude as the ~7 ms copy.
        {
            const double target = 0.006;
            nk = std::clamp<uint32_t>(uint32_t(nk * target / std::max(t_kernel, 1e-6)), 8, 4000);
            compute(); q.wait();
            c0 = now_s(); compute(); q.wait();
            t_kernel = now_s() - c0;
        }

        // This ratio is a NOISY metric: it is a difference of three wall-clock
        // timings divided by the smaller of two of them, so a single sample can
        // land optimistically high (the Phase 5 gate measured 0.972/0.981/0.981
        // where one self-reported sample had read 0.998).  Repeat it and quote
        // the SPREAD, and gate on the MEDIAN so one lucky sample cannot pass it.
        const uint32_t reps = 5;
        std::vector<double> ov, tc_v, tk_v;
        for (uint32_t r = 0; r < reps; ++r) {
            // fresh caches each rep so every copy is a real miss, never a hit
            Ds4ExpertCache c2;
            c2.init(q, arena, slots);
            const double d0 = now_s();
            { std::vector<int32_t> ids(slots);
              for (uint32_t i = 0; i < slots; ++i) ids[i] = int32_t(i + 17) % int32_t(arena_experts);
              uint32_t sl[64]; c2.acquire(2, ids.data(), slots, sl).wait(); }
            const double t_copy = now_s() - d0;

            Ds4ExpertCache c3;
            c3.init(q, arena, slots);
            const double m0 = now_s();
            { std::vector<int32_t> ids(slots);
              for (uint32_t i = 0; i < slots; ++i) ids[i] = int32_t(i + 17) % int32_t(arena_experts);
              uint32_t sl[64];
              sycl::event ev = c3.acquire(2, ids.data(), slots, sl);
              compute();
              q.wait(); ev.wait(); }
            const double t_conc = now_s() - m0;

            const double serial = t_kernel + t_copy;
            ov.push_back((serial - t_conc) / std::max(std::min(t_kernel, t_copy), 1e-12));
            tc_v.push_back(t_copy);
            tk_v.push_back(t_kernel);
        }
        (void)copy_only;

        std::vector<double> sorted = ov;
        std::sort(sorted.begin(), sorted.end());
        const double o_min = sorted.front(), o_max = sorted.back();
        const double o_med = sorted[sorted.size() / 2];
        std::printf("    %u launches | kernel alone %.3f ms | copy alone %.3f ms (rep 0)\n",
                    nk, tk_v[0] * 1e3, tc_v[0] * 1e3);
        std::printf("    per-rep overlap:");
        for (double v : ov) std::printf(" %.3f", v);
        std::printf("\n");
        std::printf("    %sMEASURED OVERLAP FRACTION : median %.3f  (range %.3f-%.3f over %u reps)%s\n"
                    "    (1.0 = the shorter side is fully hidden; >1.0 is timer noise.  Quote the\n"
                    "     RANGE, not the best sample -- this metric does not reproduce to 3 digits.)\n",
                    Y, o_med, o_min, o_max, reps, Z);
        ok("DMA and compute genuinely overlap (MEDIAN fraction > 0.5)", o_med > 0.5,
           "median " + std::to_string(o_med) + " range " + std::to_string(o_min) +
               "-" + std::to_string(o_max));
        sycl::free(wf16, q); sycl::free(xf32, q); sycl::free(yf32, q);
    }

    // Hit rate + eviction policy under a uniform-IID routing trace — the
    // `noaux_tc` worst case and the honest default.
    {
        Ds4ExpertCache c;
        c.init(q, arena, slots);
        std::mt19937 rng(1234);
        std::uniform_int_distribution<int32_t> pick(0, int32_t(arena_experts) - 1);
        const uint32_t tokens = 200, K = 6;
        for (uint32_t t = 0; t < tokens; ++t)
            for (uint32_t l = 0; l < arena_layers; ++l) {
                int32_t ids[8]; uint32_t sl[8];
                for (uint32_t k = 0; k < K; ++k) ids[k] = pick(rng);
                c.acquire(l, ids, K, sl).wait();
            }
        const auto& s = c.stats();
        const double hr  = double(s.hits) / double(s.hits + s.misses);
        const double res = double(slots) / double(arena_experts);
        std::printf("    %u tokens x %u layers x %u experts: hit rate %.1f%% at %.1f%% residency\n",
                    tokens, arena_layers, K, 100 * hr, 100 * res);
        // Under IID-uniform selection a full cache hits at ~residency.  Anything
        // far BELOW residency would mean the policy is evicting what it just
        // fetched (the LRU cyclic-flooding failure the design rejects).
        ok("hit rate is at least 80% of residency (no self-eviction)", hr >= 0.8 * res,
           std::to_string(100 * hr) + "% vs " + std::to_string(100 * res) + "%");
    }

    // Idle-time-only speculation must never evict a slot the current layer uses.
    {
        Ds4ExpertCache c;
        c.init(q, arena, /*slots_per_layer=*/8);
        const int32_t demand[6] = {1, 2, 3, 4, 5, 6};
        uint32_t sl[6];
        c.acquire(0, demand, 6, sl).wait();
        const int32_t spec[8] = {20, 21, 22, 23, 24, 25, 26, 27};
        c.speculate(0, spec, 8);
        c.transfer_queue().wait();
        bool kept = true;
        for (uint32_t k = 0; k < 6; ++k) kept = kept && c.tag(0, sl[k]) == demand[k];
        ok("speculation never evicts an in-use slot", kept);
        const auto& s = c.stats();
        ok("speculation was accounted separately", s.spec_issued > 0,
           std::to_string(s.spec_issued) + " speculative fetches");
    }

    // -----------------------------------------------------------------------
    std::printf("\n[4] Forward assembly vs the hooked reference (parity2 blobs)\n");
    // -----------------------------------------------------------------------
    const std::string P2 = dir_or("DS4_PARITY2_DIR",
                                  "${XDG_CACHE_HOME:-$HOME/.cache}/ie-deepseek4-parity/parity2");
    {
        std::ifstream mf(P2 + "/manifest.json");
        if (!mf) {
            std::printf("  %sBLOBS ABSENT at %s — §4 could not run.%s\n", R, P2.c_str(), Z);
            std::printf("  %sThis is a GAP, not a pass. Regenerate with "
                        "tools/deepseek4/parity_dump2.py --layers 4%s\n", R, Z);
            ++g_fail;
        } else {
            json j; mf >> j;
            std::map<std::string, Component> C;
            for (const auto& c : j.at("components")) {
                if (c.contains("error")) continue;
                Component comp;
                comp.name = c.at("component").get<std::string>();
                comp.cls  = c.contains("class") ? c.at("class").get<std::string>() : std::string();
                for (const auto& kv : c.at("params").items())  comp.params[kv.key()]  = parse_blob(kv.value());
                for (const auto& kv : c.at("inputs").items())  comp.inputs[kv.key()]  = parse_blob(kv.value());
                for (const auto& kv : c.at("outputs").items()) comp.outputs[kv.key()] = parse_blob(kv.value());
                C[comp.name] = std::move(comp);
            }
            auto LD = [&](const Blob& b) { return load_f32(P2 + "/" + b.file, b.numel); };
            const auto& tc = j.at("preserved_from_full");
            const uint32_t hc  = tc.at("hc_mult").get<uint32_t>();
            const float rms_e  = tc.at("rms_norm_eps").get<float>();
            const float hc_e   = tc.at("hc_eps").get<float>();
            const uint32_t nL  = j.at("tiny_config").at("num_hidden_layers").get<uint32_t>();
            const uint32_t hid = j.at("tiny_config").at("hidden_size").get<uint32_t>();
            const uint32_t T   = uint32_t(C.at("model.layers.0").inputs.at("arg_0").shape[1]);
            std::printf("  reference: %u layers, hidden=%u, hc_mult=%u, T=%u\n", nL, hid, hc, T);

            // 4a. Stream initialisation: `inputs_embeds.unsqueeze(2).expand(hc_mult)`
            //     means every stream enters layer 0 as a COPY of the embedding.
            {
                const auto s0 = LD(C.at("model.layers.0").inputs.at("arg_0"));
                double dmax = 0;
                for (uint32_t t = 0; t < T; ++t)
                    for (uint32_t m = 1; m < hc; ++m)
                        for (uint32_t d = 0; d < hid; ++d)
                            dmax = std::max(dmax, double(std::fabs(
                                s0[(size_t(t) * hc + m) * hid + d] - s0[size_t(t) * hc * hid + d])));
                ok("layer-0 streams are hc_mult identical copies (expand)", dmax == 0.0);
            }

            // 4b. The two DecoderLayer residual mixes, per layer.
            for (uint32_t L = 0; L < nL; ++L) {
                const std::string ln = "model.layers." + std::to_string(L);
                const auto st   = LD(C.at(ln).inputs.at("arg_0"));
                const auto post = LD(C.at(ln + ".attn_hc").outputs.at("out_0"));
                const auto comb = LD(C.at(ln + ".attn_hc").outputs.at("out_1"));
                const auto sub  = LD(C.at(ln + ".self_attn").outputs.at("out_0"));
                const auto exp  = LD(C.at(ln + ".ffn_hc").inputs.at("arg_0"));

                float* d_st = to_dev(q, st); float* d_po = to_dev(q, post);
                float* d_co = to_dev(q, comb); float* d_su = to_dev(q, sub);
                float* d_out = sycl::malloc_device<float>(st.size(), q);
                ds4_hc_mix(q, d_st, d_po, d_co, d_su, d_out, T, hid, hc).wait();
                const auto got = from_dev(q, d_out, st.size());
                check_tol((ln + " attn-site residual mix").c_str(), max_diff(got, exp),
                          8.0 * 5.96e-8 * std::sqrt(double(hc)) * max_abs(exp) + 1e-7);

                // FFN site, fed the reference's own attention-site output.
                const auto post2 = LD(C.at(ln + ".ffn_hc").outputs.at("out_0"));
                const auto comb2 = LD(C.at(ln + ".ffn_hc").outputs.at("out_1"));
                const auto sub2  = LD(C.at(ln + ".mlp").outputs.at("out"));
                const auto exp2  = LD(C.at(ln).outputs.at("out"));
                float* d_st2 = to_dev(q, exp); float* d_po2 = to_dev(q, post2);
                float* d_co2 = to_dev(q, comb2); float* d_su2 = to_dev(q, sub2);
                ds4_hc_mix(q, d_st2, d_po2, d_co2, d_su2, d_out, T, hid, hc).wait();
                const auto got2 = from_dev(q, d_out, st.size());
                check_tol((ln + " ffn-site  residual mix").c_str(), max_diff(got2, exp2),
                          8.0 * 5.96e-8 * std::sqrt(double(hc)) * max_abs(exp2) + 1e-7);

                // NEGATIVE CONTROL: comb NOT transposed.  Sinkhorn produces a
                // doubly-stochastic but non-symmetric matrix, so this is a real
                // mis-port that a symmetric-matrix assumption would hide.
                //
                // It MUST be run on a LATE layer.  At layer 0 all hc_mult streams
                // are identical copies of the embedding, so Σ_j comb[j,h]·s[j,d]
                // collapses to s[0,d]·(column sum) = s[0,d] and the transposed
                // form collapses to s[0,d]·(row sum) = s[0,d] as well — the two
                // are indistinguishable there BECAUSE comb is doubly stochastic.
                // Running the control at layer 0 would report "no teeth" and
                // silently retire a check that does have them.
                if (L == nL - 1) {
                    std::vector<float> combT(comb.size());
                    for (uint32_t t = 0; t < T; ++t)
                        for (uint32_t a = 0; a < hc; ++a)
                            for (uint32_t b = 0; b < hc; ++b)
                                combT[(size_t(t) * hc + a) * hc + b] =
                                    comb[(size_t(t) * hc + b) * hc + a];
                    float* d_ct = to_dev(q, combT);
                    ds4_hc_mix(q, d_st, d_po, d_ct, d_su, d_out, T, hid, hc).wait();
                    const auto bad = from_dev(q, d_out, st.size());
                    const double dev  = max_diff(bad, exp);
                    const double bnd  = 8.0 * 5.96e-8 * std::sqrt(double(hc)) * max_abs(exp) + 1e-7;
                    std::printf("  %sneg: comb consumed untransposed (layer %u)%s deviation=%.3e "
                                "= %.0fx the tolerance\n", Y, L, Z, dev, dev / bnd);
                    ok("  ...that negative control discriminates", dev > 100.0 * bnd);
                    sycl::free(d_ct, q);
                }
                for (float* p : {d_st, d_po, d_co, d_su, d_out, d_st2, d_po2, d_co2, d_su2})
                    sycl::free(p, q);
            }

            // 4c. Layer chaining — layer L's output IS layer L+1's input.
            {
                bool chained = true;
                for (uint32_t L = 0; L + 1 < nL; ++L) {
                    const auto a = LD(C.at("model.layers." + std::to_string(L)).outputs.at("out"));
                    const auto b = LD(C.at("model.layers." + std::to_string(L + 1)).inputs.at("arg_0"));
                    chained = chained && max_diff(a, b) == 0.0;
                }
                const auto a = LD(C.at("model.layers." + std::to_string(nL - 1)).outputs.at("out"));
                const auto b = LD(C.at("model.hc_head").inputs.at("arg_0"));
                chained = chained && max_diff(a, b) == 0.0;
                ok("layer L out == layer L+1 in, and last out == hc_head in", chained);
            }

            // 4d. HyperHead — the final collapse before the shared RMSNorm.
            {
                const auto& hh = C.at("model.hc_head");
                const auto x   = LD(hh.inputs.at("arg_0"));
                const auto fn  = LD(hh.params.at("hc_fn"));
                const auto bs  = LD(hh.params.at("hc_base"));
                const auto sc  = LD(hh.params.at("hc_scale"));
                const auto exp = LD(hh.outputs.at("out"));
                float* d_x = to_dev(q, x); float* d_f = to_dev(q, fn);
                float* d_b = to_dev(q, bs); float* d_s = to_dev(q, sc);
                float* d_y = sycl::malloc_device<float>(exp.size(), q);
                ds4_hyper_head(q, d_x, d_f, d_b, d_s, d_y, T, hid, hc, rms_e, hc_e).wait();
                const auto got = from_dev(q, d_y, exp.size());
                check_tol("hc_head (HyperHead stream collapse)", max_diff(got, exp),
                          8.0 * 5.96e-8 * std::sqrt(double(hc * hid)) * max_abs(exp));
                // NEGATIVE CONTROL: forget the +hc_eps on the sigmoid gate.
                std::vector<float> zero(sc.size(), 0.f);
                float* d_z = to_dev(q, zero);
                ds4_hyper_head(q, d_x, d_f, d_b, d_z, d_y, T, hid, hc, rms_e, hc_e).wait();
                const auto bad = from_dev(q, d_y, exp.size());
                std::printf("  %sneg: hc_scale zeroed%s deviation=%.3e %shas teeth%s\n",
                            Y, Z, max_diff(bad, exp), Y, Z);
                for (float* p : {d_x, d_f, d_b, d_s, d_y, d_z}) sycl::free(p, q);
            }
        }
    }

    // -----------------------------------------------------------------------
    {
        const std::string path = dir_or("DS4_GGUF",
            "${IE_MODELS_DIR}/DeepSeek-V4-Flash-0731-GGUF/UD-Q3_K_XL/"
            "DeepSeek-V4-Flash-0731-UD-Q3_K_XL-00001-of-00004.gguf");
        // The QUANT IS NOT HARDCODED HERE.  This banner used to say
        // "UD-Q3_K_XL" unconditionally, so a $DS4_GGUF pointing at the Q8 build
        // produced a whole section of output labelled Q3.  Print what was
        // actually opened.
        std::printf("\n[5] The real DeepSeek-V4-Flash-0731 model: %s\n", path.c_str());
        GgufReader g;
        const std::string oe = g.open(path);
        if (!oe.empty()) {
            std::printf("  %sMODEL NOT OPENED: %s%s\n", Y, oe.c_str(), Z);
            std::printf("  %sCriteria 1-4 (load, forward, coherent text, tok/s) NOT DEMONSTRATED.%s\n", R, Z);
        } else {
            DeepSeek4Config cfg;
            const std::string ce = read_deepseek4_config(g, cfg);
            ok("config read", ce.empty(), ce);
            DeepSeek4Model m;
            const std::string be = m.load(g, cfg);
            ok("all tensors bound", be.empty(), be);
            std::printf("    %llu tensors bound, %u layers, %u experts\n",
                        (unsigned long long)m.n_bound(), cfg.n_layers, cfg.n_experts);

            // Measure how fast the weight bytes can actually be READ before
            // committing to a multi-hour load.  This is the number that decides
            // whether an end-to-end run is possible in a session.
            // Measure how fast the weight bytes can actually be READ.  Capped in
            // BOTH bytes and wall time — on a contended USB-2 spinning disk this
            // probe would otherwise become the longest thing in the whole gate.
            // The wall check is per 256 KB because the observed floor here is
            // tens of kB/s when another process is writing to the same spindle.
            const GgufTensorInfo* probe = m.layers()[0].ffn_gate_exps;
            const size_t probe_bytes = std::min<size_t>(size_t(probe->nbytes), 64u << 20);
            const double cap = std::getenv("DS4_PROBE_SECONDS")
                                   ? std::atof(std::getenv("DS4_PROBE_SECONDS")) : 15.0;
            volatile uint64_t sink = 0;
            size_t touched = 0;
            const double r0 = now_s();
            for (size_t i = 0; i < probe_bytes; i += 4096) {
                sink += probe->data[i];
                touched = i + 4096;
                if ((i & ((256u << 10) - 1)) == 0 && now_s() - r0 > cap) break;
            }
            const double rdt = now_s() - r0;
            const double mbs = double(touched) / 1e6 / std::max(rdt, 1e-9);
            std::printf("    cold-read probe: %.2f MB touched in %.1f s -> %s%.2f MB/s%s\n",
                        double(touched) / 1e6, rdt, mbs < 200 ? R : G, mbs, Z);
            const double est = 128.2e3 / std::max(mbs, 1e-9);
            std::printf("    => reading all 128.20 GB would take %.0f s (%.1f h) at that rate\n",
                        est, est / 3600.0);
            if (est > 3600.0)
                std::printf("  %sI/O-BOUND: an end-to-end run of the real model is hours of disk "
                            "time on this box, not a code problem.%s\n", Y, Z);
            if (std::getenv("DS4_RUN_REAL")) {
                // The Phase 5 gate failed on verification coverage: this block
                // previously did ONE 4-token forward and printed a min/max.
                // Criteria 1, 2 and 4 need more than that, so it now reports
                // measured VRAM + host residency (1), checks the logits are
                // finite AND non-degenerate AND deterministic (2), and measures
                // prefill and decode tok/s with the DMA duty cycle (4).
                const HostMem before = host_mem();
                Ds4Options opt;
                opt.device_ordinal = std::getenv("DS4_GPU") ? uint32_t(std::atoi(std::getenv("DS4_GPU"))) : 1;
                opt.max_seq     = 256;
                opt.max_context = 1024;
                if (const char* pl = std::getenv("DS4_PIN_LAYERS")) opt.pin_layers = uint32_t(std::atoi(pl));

                // THE TWO-CARD LOAD.  Under $DS4_TP_GPUS the model is driven by
                // `DeepSeek4TpRuntime`, which loads one `DeepSeek4Runtime` per
                // named GPU, each holding `expert_ffn / n_cards` of every routed
                // expert.  The residency plan therefore comes out of
                // `ds4_plan_residency_tp` (via DeepSeek4Runtime::load), which
                // halves the per-card expert bytes and divides the whole-box
                // pinned-host cap by n_cards — i.e. BOTH cards' VRAM is filled
                // before anything spills.  Without it this is the single-card
                // path, unchanged.
                DeepSeek4Runtime   rtm;
                DeepSeek4TpRuntime tprt;
                FwdDriver drv;
                std::string le;
                const double l0 = now_s();
                if (tp_mode) {
                    le  = tprt.load(m, tp_options(opt, tp_gpus));
                    if (le.empty()) drv.tp = &tprt;
                } else {
                    le  = rtm.load(m, opt);
                    if (le.empty()) drv.single = &rtm;
                }
                const double load_s = now_s() - l0;
                ok(tp_mode ? "REAL MODEL LOADS ACROSS BOTH CARDS (criterion 1)"
                           : "REAL MODEL LOADS (criterion 1)", le.empty(), le);
                std::printf("    load: %s (%.1f s = %.2f h)\n",
                            le.empty() ? "OK" : le.c_str(), load_s, load_s / 3600.0);
                if (le.empty()) {
                    const HostMem after = host_mem();
                    // Criterion 1: report what was ACTUALLY resident, both sides.
                    // Under TP the per-card breakdown is the whole point, so it
                    // is printed card by card and then totalled.
                    if (tp_mode) {
                        ok("every card holds device bytes (neither GPU sat empty)",
                           report_cards(tprt, cfg.n_experts, cfg.expert_ffn));
                    }
                    DeepSeek4Runtime& c0 = drv.card0();
                    uint64_t host_all = 0, vram_all = 0, max_seg = 0;
                    for (uint32_t c = 0; c < drv.n_cards(); ++c) {
                        DeepSeek4Runtime& rc = tp_mode ? tprt.card(c) : rtm;
                        host_all += rc.arena().total_bytes();
                        vram_all += rc.resident_bytes() + rc.cache().device_bytes();
                        max_seg   = std::max(max_seg, rc.arena().max_segment_bytes());
                    }
                    std::printf("    %sVRAM %.3f GB over %u card(s) | pinned host arena %.3f GB"
                                " (largest segment %.3f GB, cap %.3f GB)%s\n", G,
                                double(vram_all) / 1e9, drv.n_cards(),
                                double(host_all) / 1e9, double(max_seg) / 1e9,
                                double(kDs4MaxAllocBytes) / 1e9, Z);
                    std::printf("    host RSS %.2f -> %.2f GB (delta %.2f GB) | MemAvailable %.2f "
                                "-> %.2f GB | expert cache %.3f GB VRAM, %u slots/layer (%.1f%% "
                                "residency)\n",
                                before.rss_gb, after.rss_gb, after.rss_gb - before.rss_gb,
                                before.avail_gb, after.avail_gb,
                                double(c0.cache().device_bytes()) / 1e9,
                                c0.cache().slots_per_layer(),
                                100.0 * c0.cache().slots_per_layer() / cfg.n_experts);
                    ok("every segment is within the 32.53 GB per-allocation cap (criterion 5)",
                       max_seg <= kDs4MaxAllocBytes, std::to_string(max_seg));
                    if (c0.layers_pinned() != cfg.n_layers)
                        std::printf("  %sPARTIAL PIN: only %u of %u layers pinned (DS4_PIN_LAYERS) "
                                    "-- criterion 1 is NOT fully demonstrated by this run.%s\n",
                                    Y, c0.layers_pinned(), cfg.n_layers, Z);

                    // ---- prefill (criterion 2 + criterion 4 prefill tok/s) ----
                    const uint32_t PT = 128;
                    std::vector<int32_t> prompt(PT);
                    for (uint32_t i = 0; i < PT; ++i) prompt[i] = int32_t((i * 131 + 7) % cfg.vocab);
                    std::vector<float> lg(cfg.vocab), lg2(cfg.vocab);
                    const double p0 = now_s();
                    const std::string fe = drv.forward(prompt.data(), PT, 0, lg.data());
                    const double p_s = now_s() - p0;
                    ok(tp_mode ? "REAL FORWARD PASS RUNS ACROSS BOTH CARDS (criterion 2)"
                               : "REAL FORWARD PASS RUNS (criterion 2)", fe.empty(), fe);
                    if (fe.empty()) {
                        bool fin = true, allzero = true;
                        double mn = 1e30, mx = -1e30, sum = 0;
                        for (float v : lg) {
                            fin = fin && std::isfinite(v);
                            if (v != 0.f) allzero = false;
                            mn = std::min(mn, double(v)); mx = std::max(mx, double(v)); sum += v;
                        }
                        ok("real logits are all finite (no NaN / Inf)", fin);
                        ok("real logits are not degenerate (not all zero)", !allzero);
                        ok("real logits have spread (max-min > 1e-3)", mx - mn > 1e-3,
                           "range " + std::to_string(mx - mn));
                        std::printf("    logits: min=%.4f max=%.4f mean=%.4f range=%.4f\n",
                                    mn, mx, sum / cfg.vocab, mx - mn);
                        const auto& tm = drv.card0().last_timing();
                        std::printf("    %sPREFILL: %u tok in %.1f ms -> %.2f tok/s%s | DMA %.3f GB, "
                                    "queue busy %.1f ms (%.0f%% of wall), %llu hit / %llu miss\n",
                                    Y, PT, p_s * 1e3, double(PT) / p_s, Z,
                                    double(tm.bytes_fetched) / 1e9, tm.dma_seconds * 1e3,
                                    100.0 * tm.dma_seconds / std::max(tm.wall_seconds, 1e-9),
                                    (unsigned long long)tm.hits, (unsigned long long)tm.misses);

                        // Determinism: the same prompt from a reset context must
                        // reproduce bit-for-bit, or nothing measured here means
                        // anything.
                        drv.reset_context();
                        const std::string fe2 = drv.forward(prompt.data(), PT, 0, lg2.data());
                        bool same = fe2.empty();
                        if (same) for (uint32_t v = 0; v < cfg.vocab; ++v)
                            if (lg[v] != lg2[v]) { same = false; break; }
                        ok("real logits are bit-deterministic across runs", same, fe2);
                    }

                    // ---- decode: greedy steps, tok/s + DMA duty (criterion 4) ----
                    if (fe.empty()) {
                        int32_t tok = 0;
                        { double best = -1e30; for (uint32_t v = 0; v < cfg.vocab; ++v)
                              if (lg[v] > best) { best = lg[v]; tok = int32_t(v); } }
                        const uint32_t steps = std::getenv("DS4_DECODE_STEPS")
                                                   ? uint32_t(std::atoi(std::getenv("DS4_DECODE_STEPS"))) : 8;
                        double dec_s = 0, dma_s = 0, wall_s = 0;
                        uint64_t bytes = 0, hits = 0, misses = 0;
                        bool all_ok = true, all_fin = true;
                        std::vector<int32_t> produced;
                        for (uint32_t s = 0; s < steps; ++s) {
                            const double d0 = now_s();
                            const std::string de = drv.forward(&tok, 1, PT + s, lg.data());
                            dec_s += now_s() - d0;
                            if (!de.empty()) { all_ok = false;
                                std::printf("    decode step %u: %s\n", s, de.c_str()); break; }
                            double best = -1e30;
                            for (uint32_t v = 0; v < cfg.vocab; ++v) {
                                if (!std::isfinite(lg[v])) all_fin = false;
                                if (lg[v] > best) { best = lg[v]; tok = int32_t(v); }
                            }
                            produced.push_back(tok);
                            const auto& tm = drv.card0().last_timing();
                            dma_s += tm.dma_seconds; wall_s += tm.wall_seconds;
                            bytes += tm.bytes_fetched; hits += tm.hits; misses += tm.misses;
                        }
                        ok("real decode steps completed", all_ok);
                        ok("every real decode step produced finite logits", all_fin);
                        if (all_ok) {
                            std::printf("    %sDECODE: %.1f ms/token -> %.3f tok/s%s over %u steps\n",
                                        Y, dec_s / steps * 1e3, double(steps) / dec_s, Z, steps);
                            std::printf("    decode DMA: %.1f MB/token, queue busy %.1f ms/token = "
                                        "%s%.0f%% DMA duty cycle%s, hit rate %.1f%%\n",
                                        double(bytes) / steps / 1e6, dma_s / steps * 1e3, Y,
                                        100.0 * dma_s / std::max(wall_s, 1e-9), Z,
                                        100.0 * double(hits) / double(std::max<uint64_t>(hits + misses, 1)));
                            std::printf("    greedy token ids:");
                            for (int32_t t : produced) std::printf(" %d", t);
                            std::printf("\n");
                            // Phase 6 shipped the tokenizer, so these ids no longer
                            // have to stay ids.  They are STILL not evidence of
                            // coherent English: the prompt above is 128 ARBITRARY
                            // ids picked by (i*131+7) % vocab, i.e. not text, so a
                            // sane model's continuation of it is not text either.
                            // The real-prompt run below is the one that can speak.
                            {
                                Tokenizer tk;
                                if (auto te = tk.load_from_gguf(g); te.empty()) {
                                    const std::string txt = tk.decode(
                                        std::span<const int32_t>(produced), /*skip_special=*/false,
                                        std::span<const int32_t>{});
                                    std::printf("    detokenized (from the ARBITRARY-id prompt): \"%s\"\n",
                                                printable(txt).c_str());
                                } else {
                                    std::printf("    %stokenizer did not load: %s%s\n", R, te.c_str(), Z);
                                }
                            }
                            if (tp_mode) {
                                // The reduction count is the proof the cards were
                                // actually driven in lockstep: one per layer per
                                // forward, no more and no less.
                                const uint64_t want =
                                    uint64_t(cfg.n_layers) * (2 + steps);   // prefill x2 + decode
                                std::printf("    %sTP: %llu cross-card reductions in %.1f ms"
                                            " (expected %llu = %u layers x (2 prefill + %u decode))"
                                            " | staging %llu B pinned%s\n", Y,
                                            (unsigned long long)tprt.reductions(),
                                            tprt.reduce_seconds() * 1e3,
                                            (unsigned long long)want, cfg.n_layers, steps,
                                            (unsigned long long)tprt.stage_bytes(), Z);
                                ok("one cross-card reduction per layer per forward",
                                   tprt.reductions() == want,
                                   std::to_string(tprt.reductions()) + " vs " + std::to_string(want));
                            }
                        }
                    }

                    // ---- THE ONE THING NO RUN HAS EVER DONE: read the output ----
                    // Everything above is finite/deterministic/fast.  None of it
                    // can tell a correct model from a subtly wrong one — a
                    // permuted expert, a mis-scaled RoPE and a bit-exact forward
                    // all produce finite logits with spread.  Only a REAL prompt,
                    // decoded back to text, can.  So: encode an actual chat turn
                    // with the model's own tokenizer, prefill it, greedy-decode,
                    // and PRINT WHAT IT SAID.  $DS4_TEXT_TOKENS sets the budget
                    // (0 skips this block entirely).
                    const uint32_t text_budget = std::getenv("DS4_TEXT_TOKENS")
                        ? uint32_t(std::atoi(std::getenv("DS4_TEXT_TOKENS"))) : 48u;
                    Tokenizer tk;
                    const std::string tke = tk.load_from_gguf(g);
                    ok("the real model's tokenizer loads from its own GGUF", tke.empty(), tke);
                    if (fe.empty() && text_budget > 0 && tke.empty()) {
                        const ChatTurn turns[] = {
                            {"user", std::getenv("DS4_TEXT_PROMPT")
                                         ? std::getenv("DS4_TEXT_PROMPT")
                                         : "Explain in two sentences why the sky is blue."}};
                        DeepSeek4ChatOptions copt;
                        copt.thinking = false;
                        const std::string ptxt = build_deepseek4_prompt(
                            std::span<const ChatTurn>(turns, 1), /*add_generation_prompt=*/true, copt);
                        std::vector<int32_t> pids = tk.encode(ptxt, /*allow_special=*/true);
                        std::printf("    real prompt: %zu tokens\n      %s\n",
                                    pids.size(), printable(ptxt).c_str());
                        ok("the chat prompt encodes to a non-empty id list", !pids.empty(),
                           std::to_string(pids.size()));

                        drv.reset_context();
                        std::string ferr;
                        uint32_t p = 0;
                        const uint32_t chunk = 128;   // opt.max_seq above
                        const double g0 = now_s();
                        while (p < pids.size() && ferr.empty()) {
                            const uint32_t n = std::min<uint32_t>(chunk, uint32_t(pids.size()) - p);
                            ferr = drv.forward(pids.data() + p, n, p, lg.data());
                            p += n;
                        }
                        const double gp = now_s() - g0;
                        std::vector<int32_t> gen;
                        const int32_t eos = tk.eos_token_id();
                        const double d0 = now_s();
                        for (uint32_t s = 0; s < text_budget && ferr.empty(); ++s) {
                            int32_t t = 0; double best = -1e30;
                            for (uint32_t v = 0; v < cfg.vocab; ++v)
                                if (lg[v] > best) { best = lg[v]; t = int32_t(v); }
                            if (t == eos) break;
                            gen.push_back(t);
                            ferr = drv.forward(&t, 1, p, lg.data());
                            ++p;
                        }
                        const double dd = now_s() - d0;
                        ok("the real-prompt generation ran", ferr.empty(), ferr);
                        const std::string said =
                            tk.decode(std::span<const int32_t>(gen), /*skip_special=*/true,
                                      std::span<const int32_t>{});
                        std::printf("    prefill %zu tok in %.1f s (%.2f tok/s) | %zu new tok in"
                                    " %.1f s (%.3f tok/s)\n",
                                    pids.size(), gp, double(pids.size()) / std::max(gp, 1e-9),
                                    gen.size(), dd, double(gen.size()) / std::max(dd, 1e-9));
                        std::printf("    %s================ DeepSeek-V4-Flash SAID ================%s\n", G, Z);
                        std::printf("%s\n", said.c_str());
                        std::printf("    %s=======================================================%s\n", G, Z);
                        std::printf("    %sJUDGE THIS BY EYE. Finite logits proved nothing about"
                                    " correctness; readable, on-topic\n"
                                    "    English is the first evidence the numerical stack is"
                                    " right. Garbage here is a\n"
                                    "    CRITICAL defect, not a rough edge.%s\n", Y, Z);
                    } else if (text_budget == 0) {
                        std::printf("    %sDS4_TEXT_TOKENS=0: the real-prompt text generation was"
                                    " SKIPPED -- coherent English is NOT demonstrated.%s\n", Y, Z);
                    }
                }
            } else {
                std::printf("  %sDS4_RUN_REAL not set: the end-to-end load/decode was NOT attempted.%s\n", Y, Z);
                std::printf("  %sCriteria 1-4 remain UNPROVEN by this run.%s\n", Y, Z);
            }
        }
    }

    // -----------------------------------------------------------------------
    std::printf("\n[6] End-to-end forward on a miniature but structurally real deepseek4 GGUF\n");
    // -----------------------------------------------------------------------
    // Carried into §7 only so §7's own single-card run of the SAME fixture can be
    // cross-checked against it; §6's arithmetic is untouched.
    std::vector<int32_t> s6_greedy;
    {
        const std::string tmp = dir_or("TMPDIR", "/tmp") + "/ds4_mini.gguf";
        mini::Cfg mc;
        std::vector<std::vector<uint8_t>> keep;
        const std::string we = mini::write_gguf(mc, tmp, keep);
        ok("wrote a miniature deepseek4 GGUF", we.empty(), we);
        keep.clear();

        GgufReader g;
        const std::string oe = g.open(tmp);
        ok("engine's own GgufReader parses it", oe.empty(), oe);
        DeepSeek4Config cfg;
        const std::string ce = read_deepseek4_config(g, cfg);
        ok("read_deepseek4_config", ce.empty(), ce);
        DeepSeek4Model m;
        const std::string be = m.load(g, cfg);
        ok("DeepSeek4Model binds every tensor (same 1:1 contract)", be.empty(), be);
        std::printf("    %llu tensors, %u layers (%u hash / sliding+CSA+HCA), %u experts, hidden=%u\n",
                    (unsigned long long)m.n_bound(), cfg.n_layers, cfg.hash_layer_count,
                    cfg.n_experts, cfg.hidden);

        Ds4Options opt;
        opt.device_ordinal = std::getenv("DS4_GPU") ? uint32_t(std::atoi(std::getenv("DS4_GPU"))) : 1;
        // Long enough that BOTH compressors actually emit: the ratio-4 (CSA)
        // one fires after 4 tokens, but the ratio-128 (HCA) one needs 128, and
        // the sliding ring (window 16) must evict.  A 16-token prompt would
        // leave the HCA pooling path and the ring-offset mask arithmetic
        // completely unexercised.
        opt.max_seq        = 192;
        opt.max_context    = 512;
        opt.slots_per_layer = 4;              // < n_experts, so the cache really misses
        opt.expert_cache_bytes = 64ull << 20;
        // Captured so §7 can compare the two-card run against THIS run's own
        // numbers rather than against a remembered constant.  Pure host copies
        // (plus one D2H read of the accumulator, taken only under $DS4_TP_GPUS);
        // the single-card arithmetic below is untouched either way.
        DeepSeek4Runtime rtm;
        const double l0 = now_s();
        const std::string le = rtm.load(m, opt);
        const double load_s = now_s() - l0;
        ok("DeepSeek4Runtime::load (resident upload + pinned pool + cache)", le.empty(), le);
        if (le.empty()) {
            std::printf("    load %.2f s | resident %.1f MB VRAM | pinned pool %.1f MB in %zu segments"
                        " | %u slots/layer\n",
                        load_s, double(rtm.resident_bytes()) / 1e6,
                        double(rtm.arena().total_bytes()) / 1e6, rtm.arena().n_segments(),
                        rtm.cache().slots_per_layer());

            // ---- prefill ----
            std::vector<int32_t> prompt(136);
            for (uint32_t i = 0; i < prompt.size(); ++i) prompt[i] = int32_t((i * 37 + 5) % cfg.vocab);
            std::vector<float> lg(cfg.vocab);
            const double p0 = now_s();
            const std::string fe = rtm.forward(prompt.data(), uint32_t(prompt.size()), 0, lg.data());
            const double p_s = now_s() - p0;
            ok("prefill 136 tok (sliding ring evicts, CSA + HCA both emit)", fe.empty(), fe);
            if (fe.empty()) {
                bool fin = true, allzero = true;
                double mn = 1e30, mx = -1e30, sum = 0;
                for (float v : lg) {
                    fin = fin && std::isfinite(v);
                    if (v != 0.f) allzero = false;
                    mn = std::min(mn, double(v)); mx = std::max(mx, double(v)); sum += v;
                }
                ok("logits are all finite (no NaN / Inf)", fin);
                ok("logits are not degenerate (not all zero)", !allzero);
                ok("logits have spread (max-min > 1e-3)", mx - mn > 1e-3,
                   "range " + std::to_string(mx - mn));
                std::printf("    logits: min=%.4f max=%.4f mean=%.4f | prefill %.1f ms "
                            "(%.1f tok/s)\n", mn, mx, sum / cfg.vocab, p_s * 1e3,
                            double(prompt.size()) / p_s);
                const auto& tm = rtm.last_timing();
                std::printf("    prefill DMA: %.3f MB fetched, transfer queue busy %.1f ms "
                            "(%.0f%% of wall), %llu hit / %llu miss\n",
                            double(tm.bytes_fetched) / 1e6, tm.dma_seconds * 1e3,
                            100.0 * tm.dma_seconds / std::max(tm.wall_seconds, 1e-9),
                            (unsigned long long)tm.hits, (unsigned long long)tm.misses);
            }

            // ---- decode: 8 greedy steps, measuring tok/s and overlap ----
            if (fe.empty()) {
                int32_t tok = 0;
                { double best = -1e30; for (uint32_t v = 0; v < cfg.vocab; ++v)
                      if (lg[v] > best) { best = lg[v]; tok = int32_t(v); } }
                const uint32_t steps = 8;
                double dec_s = 0, dma_s = 0, wall_s = 0;
                uint64_t bytes = 0, hits = 0, misses = 0;
                bool all_ok = true, all_fin = true;
                std::vector<int32_t> produced;
                for (uint32_t s = 0; s < steps; ++s) {
                    const double d0 = now_s();
                    const std::string de = rtm.forward(&tok, 1, uint32_t(prompt.size() + s), lg.data());
                    dec_s += now_s() - d0;
                    if (!de.empty()) { all_ok = false; std::printf("    decode step %u: %s\n", s, de.c_str()); break; }
                    double best = -1e30;
                    for (uint32_t v = 0; v < cfg.vocab; ++v) {
                        if (!std::isfinite(lg[v])) all_fin = false;
                        if (lg[v] > best) { best = lg[v]; tok = int32_t(v); }
                    }
                    produced.push_back(tok);
                    const auto& tm = rtm.last_timing();
                    dma_s += tm.dma_seconds; wall_s += tm.wall_seconds;
                    bytes += tm.bytes_fetched; hits += tm.hits; misses += tm.misses;
                }
                ok("8 greedy decode steps completed", all_ok);
                ok("every decode step produced finite logits", all_fin);
                if (all_ok) s6_greedy = produced;
                if (all_ok) {
                    std::printf("    decode: %.1f ms/token -> %s%.2f tok/s%s\n",
                                dec_s / steps * 1e3, Y, double(steps) / dec_s, Z);
                    std::printf("    decode DMA: %.1f MB/token, transfer queue busy %.1f ms/token "
                                "= %s%.0f%% DMA duty cycle%s, hit rate %.1f%%\n",
                                double(bytes) / steps / 1e6, dma_s / steps * 1e3, Y,
                                100.0 * dma_s / std::max(wall_s, 1e-9), Z,
                                100.0 * double(hits) / double(std::max<uint64_t>(hits + misses, 1)));
                    std::printf("    greedy token ids: ");
                    for (int32_t t : produced) std::printf("%d ", t);
                    std::printf("\n    %s(random weights -> these ids carry no linguistic meaning; "
                                "COHERENT TEXT IS NOT DEMONSTRATED)%s\n", Y, Z);
                }
            }

            // Criterion 5, on the path a real run would take: a chunk request
            // above the device cap must fail loudly, not silently downsize.
            {
                Ds4HostArena a2;
                const std::string se = a2.init(rtm.queue(), std::vector<uint64_t>{1ull << 20},
                                               8, kDs4MaxAllocBytes * 2);
                ok("runtime arena still refuses an over-cap segment", !se.empty());
            }
        }
        std::remove(tmp.c_str());
    }

    // -----------------------------------------------------------------------
    // §7 — BOTH GPUs: the two-card orchestrator on the miniature fixture
    // -----------------------------------------------------------------------
    // Runs only under $DS4_TP_GPUS with more than one ordinal.  Unlike §5 (where
    // the real 128.20 GB model can only be loaded ONCE, so TP REPLACES the
    // single-card load), the miniature model is cheap enough to load both ways
    // in one process, and it has to be: hidden-dim expert-TP is a PARTITION of
    // the intermediate dimension, so the claim under test is that the two-card
    // run reproduces the single-card one.  A two-card number with no single-card
    // reference beside it proves nothing.
    //
    // WHY TWO FIXTURES.  The split's ONLY lossy step is that each card's sliced
    // `down` GEMV writes an fp16 partial where one card writes one fp16 whole
    // (deepseek4_residency_test §15a measures that at <= 2 ulps).  Whether a
    // 2-ulp seam shows up as "identical" or "unrecognisable" 6 layers later is a
    // property of the FIXTURE, not of the split, so §7 runs the identical
    // comparison on two fixtures that differ in exactly ONE knob — the routed
    // experts' IQ3_XXS super-scale — and reports both.  That is a controlled
    // experiment, not an excuse: if the conditioned fixture also diverged, the
    // split would be the suspect and this section would say so.
    if (tp_mode) {
        std::printf("\n[7] Two-card (%zu-GPU) load + forward, against the single-card run\n",
                    tp_gpus.size());

        struct Cmp {
            bool     ran = false;
            bool     tokens_match = false;
            double   moe_rel = 0, lg_rel = 0, moe_scale = 0, half_ratio = 0;
            std::vector<int32_t> greedy1, greedy2;
        };

        // One fixture, end to end: write it, run it on ONE card, run it on ALL
        // the named cards, compare.  `gate` says whether the comparison counts
        // as a failure — §7b's fixture is reported but cannot discriminate, and
        // the run below is what establishes that rather than a claim.
        auto run_fixture = [&](const mini::Cfg& mc, const char* label, bool gate,
                               Cmp& out) {
            std::printf("\n  %s%s%s\n", Y, label, Z);
            const std::string path = dir_or("TMPDIR", "/tmp") + "/ds4_tp_fwd_" +
                                     std::to_string(mc.iq3_scale) + ".gguf";
            std::vector<std::vector<uint8_t>> keep;
            const std::string we = mini::write_gguf(mc, path, keep);
            ok("wrote the fixture", we.empty(), we);
            keep.clear();
            if (!we.empty()) return;

            GgufReader g2;
            const std::string oe = g2.open(path);
            ok("GgufReader parses it", oe.empty(), oe);
            DeepSeek4Config cfg2;
            const std::string ce2 = oe.empty() ? read_deepseek4_config(g2, cfg2) : std::string("skipped");
            ok("read_deepseek4_config", ce2.empty(), ce2);
            DeepSeek4Model m2;
            const std::string be2 = ce2.empty() ? m2.load(g2, cfg2) : std::string("skipped");
            ok("DeepSeek4Model binds every tensor", be2.empty(), be2);
            if (!oe.empty() || !ce2.empty() || !be2.empty()) { std::remove(path.c_str()); return; }

            Ds4Options o;
            o.device_ordinal      = std::getenv("DS4_GPU") ? uint32_t(std::atoi(std::getenv("DS4_GPU"))) : 1;
            o.max_seq             = 192;
            o.max_context         = 512;
            o.slots_per_layer     = 4;
            o.expert_cache_bytes  = 64ull << 20;
            o.prefetch            = false;   // speculation is timing-dependent; keep the two
                                             // paths' cache traffic comparable

            std::vector<int32_t> prompt(136);
            for (uint32_t i = 0; i < prompt.size(); ++i) prompt[i] = int32_t((i * 37 + 5) % cfg2.vocab);
            const uint32_t STEPS = 8;

            // ---- one card ----
            std::vector<float> lg1(cfg2.vocab), moe1;
            {
                DeepSeek4Runtime r1;
                const std::string le1 = r1.load(m2, o);
                ok("single-card load", le1.empty(), le1);
                if (!le1.empty()) { std::remove(path.c_str()); return; }
                ok("the single-card runtime holds the WHOLE expert (n_cards == 1)",
                   r1.expert_slice().n_cards == 1 && r1.expert_slice().efc == cfg2.expert_ffn);
                const std::string f1 = r1.forward(prompt.data(), uint32_t(prompt.size()), 0, lg1.data());
                ok("single-card prefill", f1.empty(), f1);
                if (!f1.empty()) { std::remove(path.c_str()); return; }
                moe1.assign(size_t(prompt.size()) * cfg2.hidden, 0.f);
                r1.queue().memcpy(moe1.data(), r1.moe_accumulator(), moe1.size() * 4).wait();
                int32_t tok = 0;
                { double best = -1e30; for (uint32_t v = 0; v < cfg2.vocab; ++v)
                      if (lg1[v] > best) { best = lg1[v]; tok = int32_t(v); } }
                std::vector<float> lg(cfg2.vocab);
                for (uint32_t s = 0; s < STEPS; ++s) {
                    const std::string de = r1.forward(&tok, 1, uint32_t(prompt.size() + s), lg.data());
                    if (!de.empty()) { ok("single-card decode", false, de); break; }
                    double best = -1e30;
                    for (uint32_t v = 0; v < cfg2.vocab; ++v)
                        if (lg[v] > best) { best = lg[v]; tok = int32_t(v); }
                    out.greedy1.push_back(tok);
                }
                ok("single-card decode completed", out.greedy1.size() == STEPS);
            }

            // ---- every named card, in lockstep ----
            DeepSeek4TpRuntime tp;
            const std::string tle = tp.load(m2, tp_options(o, tp_gpus));
            ok("DeepSeek4TpRuntime::load across the named GPUs", tle.empty(), tle);
            if (!tle.empty()) {
                std::printf("  %sThe two-card path did NOT run; nothing below it is"
                            " demonstrated.%s\n", R, Z);
                std::remove(path.c_str());
                return;
            }
            ok("every card holds device bytes (neither GPU sat empty)",
               report_cards(tp, cfg2.n_experts, cfg2.expert_ffn));

            // The PHYSICAL device, not merely the recorded ordinal: a map that
            // quietly collapsed onto one card would pass an ordinal check and
            // fail this one.
            bool bound = tp.n_cards() == tp_gpus.size();
            for (uint32_t c = 0; c < tp.n_cards(); ++c)
                bound = bound && tp.device_of(c) == tp_gpus[c];
            ok("each card's queue landed on the GPU it was assigned", bound);
            bool distinct = true;
            for (uint32_t c = 1; c < tp.n_cards(); ++c)
                distinct = distinct && tp.card(c).queue().get_device() !=
                                       tp.card(0).queue().get_device();
            ok("the cards are on PHYSICALLY DIFFERENT devices", distinct);
            ok("each card holds exactly 1/n_cards of the expert bytes",
               tp.card(0).tp_residency().card_slot_total * tp.n_cards() ==
                   tp.card(0).tp_residency().whole_slot_total,
               std::to_string(tp.card(0).tp_residency().card_slot_total) + " x " +
                   std::to_string(tp.n_cards()) + " vs " +
                   std::to_string(tp.card(0).tp_residency().whole_slot_total));
            uint32_t covered = 0;
            bool tiles = true;
            for (uint32_t c = 0; c < tp.n_cards(); ++c) {
                tiles = tiles && tp.card(c).expert_slice().ef0 == covered;
                covered += tp.card(c).expert_slice().efc;
            }
            ok("the cards' slices TILE the intermediate dimension exactly",
               tiles && covered == cfg2.expert_ffn,
               "covered [0," + std::to_string(covered) + ") of " +
                   std::to_string(cfg2.expert_ffn));

            std::vector<float> lg2v(cfg2.vocab);
            const double p0 = now_s();
            const std::string tf = tp.forward(prompt.data(), uint32_t(prompt.size()), 0, lg2v.data());
            const double tps = now_s() - p0;
            ok("two-card prefill (the same 136 tokens)", tf.empty(), tf);
            if (!tf.empty()) { std::remove(path.c_str()); return; }
            std::vector<float> moe2(moe1.size(), 0.f);
            tp.card(0).queue().memcpy(moe2.data(), tp.card(0).moe_accumulator(),
                                      moe2.size() * 4).wait();
            bool fin = true;
            for (float v : lg2v) fin = fin && std::isfinite(v);
            ok("two-card logits are all finite", fin);
            std::printf("      two-card prefill %.1f ms (%.1f tok/s -- cards run SEQUENTIALLY,"
                        " so this is NOT a TP throughput number)\n",
                        tps * 1e3, double(prompt.size()) / tps);

            {
                int32_t tok = 0;
                { double best = -1e30; for (uint32_t v = 0; v < cfg2.vocab; ++v)
                      if (lg2v[v] > best) { best = lg2v[v]; tok = int32_t(v); } }
                std::vector<float> lg(cfg2.vocab);
                for (uint32_t s = 0; s < STEPS; ++s) {
                    const std::string de = tp.forward(&tok, 1, uint32_t(prompt.size() + s), lg.data());
                    if (!de.empty()) { ok("two-card decode", false, de); break; }
                    double best = -1e30;
                    for (uint32_t v = 0; v < cfg2.vocab; ++v)
                        if (lg[v] > best) { best = lg[v]; tok = int32_t(v); }
                    out.greedy2.push_back(tok);
                }
                ok("two-card decode completed", out.greedy2.size() == STEPS);
            }

            // The forward returning "" IS the lockstep proof: every layer of
            // every call compared every card's expert ids AND routing weights
            // against card 0's, bit for bit, and would have refused the whole
            // pass on any difference.
            const uint64_t want = uint64_t(cfg2.n_layers) * (1 + uint64_t(out.greedy2.size()));
            std::printf("      %llu cross-card reductions in %.3f ms (expected %llu = %u layers"
                        " x (1 prefill + %zu decode)); %llu B pinned staging\n",
                        (unsigned long long)tp.reductions(), tp.reduce_seconds() * 1e3,
                        (unsigned long long)want, cfg2.n_layers, out.greedy2.size(),
                        (unsigned long long)tp.stage_bytes());
            ok("one cross-card reduction per layer per forward", tp.reductions() == want,
               std::to_string(tp.reductions()) + " vs " + std::to_string(want));

            // ---- the sliced runtimes still refuse on their own ----
            for (uint32_t c = 0; c < tp.n_cards(); ++c) {
                std::vector<float> junk(cfg2.vocab, 0.f);
                const std::string e1 = tp.card(c).forward(prompt.data(), 4, 0, junk.data());
                ok(("card " + std::to_string(c) + "'s own forward() refuses (PARTIAL)").c_str(),
                   !e1.empty() && e1.find("PARTIAL") != std::string::npos &&
                       e1.find("DeepSeek4TpRuntime") != std::string::npos, e1);
            }

            // ---- THE COMPARISON ----
            out.ran          = true;
            out.moe_scale    = max_abs(moe1);
            const double md  = max_diff(moe2, moe1);
            out.moe_rel      = md / std::max(out.moe_scale, 1e-30);
            out.lg_rel       = max_diff(lg2v, lg1) / std::max(max_abs(lg1), 1e-30);
            // A DROPPED reduction leaves half the routed sum behind, an error of
            // 0.5*max|ref|.  This ratio is how many times closer the real
            // two-card answer is than that; it is computed from this run's own
            // numbers, never assumed.
            out.half_ratio   = (0.5 * out.moe_scale) / std::max(md, 1e-30);
            out.tokens_match = out.greedy1.size() == STEPS && out.greedy2 == out.greedy1;
            std::printf("      routed-expert accumulator: max|1-card| = %.6g,"
                        " max|2-card - 1-card| = %.6g  (%.3e relative)\n",
                        out.moe_scale, md, out.moe_rel);
            std::printf("      prefill logits           : %.3e relative\n", out.lg_rel);
            std::printf("      vs a DROPPED reduction   : %.2fx closer\n", out.half_ratio);
            std::printf("      1-card greedy:");
            for (int32_t t : out.greedy1) std::printf(" %d", t);
            std::printf("\n      2-card greedy:");
            for (int32_t t : out.greedy2) std::printf(" %d", t);
            std::printf("\n");

            if (gate) {
                ok("the two-card routed sum is >= 10x closer to the single card's than the"
                   " half-sum a DROPPED reduction would leave",
                   out.half_ratio > 10.0, std::to_string(out.half_ratio) + "x");
                ok("the two-card greedy token sequence is IDENTICAL to the single card's",
                   out.tokens_match);
            }
            std::remove(path.c_str());
        };

        // 7a — the fixture the comparison is GATED on.  Identical to §6's in
        //      every respect except the routed experts' super-scale.
        mini::Cfg cond;
        cond.iq3_scale = 5e-4f;
        Cmp a;
        run_fixture(cond, "7a  IQ3_XXS super-scale 5e-4 -- conditioned fixture (GATED)", true, a);

        // 7b — §6's OWN fixture, byte for byte.  Reported, not gated: see below.
        Cmp b;
        run_fixture(mini::Cfg{},
                    "7b  IQ3_XXS super-scale 2e-2 -- §6's fixture (DIAGNOSTIC, not gated)",
                    false, b);

        // §7b's single-card run must reproduce §6's, or the two sections are not
        // running the same fixture and 7b's diagnosis would be about something
        // else.  THIS is gated.
        if (!s6_greedy.empty())
            ok("§7b's single-card run reproduces §6's greedy ids exactly (same fixture)",
               b.greedy1 == s6_greedy);

        // ---- what the controlled experiment says ----
        std::printf("\n  %s---- 7a vs 7b: one knob changed, the split identical ----%s\n", Y, Z);
        std::printf("    %-28s %-12s %-12s %-10s %s\n",
                    "fixture", "moe rel", "logit rel", "vs-dropped", "greedy ids");
        auto row = [&](const char* nm, const Cmp& c) {
            std::printf("    %-28s %-12.3e %-12.3e %-10.2f %s\n", nm, c.moe_rel, c.lg_rel,
                        c.half_ratio, c.ran ? (c.tokens_match ? "IDENTICAL" : "DIVERGED")
                                            : "did not run");
        };
        row("7a  super-scale 5e-4", a);
        row("7b  super-scale 2e-2", b);
        if (a.ran && b.ran) {
            if (a.tokens_match && !b.tokens_match) {
                std::printf("    %sThe SAME split reproduces the single card exactly at 5e-4 and"
                            " diverges at 2e-2.\n"
                            "    The split is therefore not what changed -- the fixture's"
                            " conditioning is.  At 2e-2\n"
                            "    this model's clamped SwiGLU sits against its clamp, so the"
                            " routed-expert block is\n"
                            "    near-discontinuous and the <=2-ulp fp16 seam"
                            " (deepseek4_residency_test §15a)\n"
                            "    lands on the far side of a clamp.  max|accumulator| = %.4g at"
                            " 2e-2 vs %.4g at 5e-4\n"
                            "    is the same saturation seen from outside.%s\n",
                            G, b.moe_scale, a.moe_scale, Z);
            } else if (!a.tokens_match) {
                std::printf("    %sThe CONDITIONED fixture diverged too.  That is NOT a"
                            " conditioning story: the split\n"
                            "    itself is the suspect and 7a's gate above has failed."
                            " Do not read 7b as an excuse.%s\n", R, Z);
            } else {
                std::printf("    %sBoth fixtures reproduce the single card exactly.%s\n", G, Z);
            }
        }
    }

    // -----------------------------------------------------------------------
    std::printf("\n[8] The miniature fixture through ie::Engine — load, generate, DETOKENIZE\n");
    // -----------------------------------------------------------------------
    // §6/§7 drive DeepSeek4Runtime directly and stop at token IDS.  This section
    // drives the SAME fixture through the product API — Engine::load (which must
    // pick the kDeepSeek4 branch), Engine::generate (prefill, sample, decode) and
    // the streaming TokenCallback — and PRINTS THE DETOKENIZED TEXT.
    //
    // WHAT IT PROVES: that the deepseek4 branch of Engine::load exists and is
    // reached, that the runtime is driven correctly through forward_step, that
    // the fp32→fp16 logit bounce feeds the shared GPU sampler, and that ids come
    // back out through Tokenizer::decode as text.
    // WHAT IT DOES NOT PROVE: anything about DeepSeek's trained weights.  The
    // fixture's weights are pseudo-random, so the TEXT IS MEANINGLESS BY
    // CONSTRUCTION — the round trip is what is under test, not the language.
    {
        const std::string base = dir_or("TMPDIR", "/tmp") + "/ds4_mini_engine_base.gguf";
        const std::string path = dir_or("TMPDIR", "/tmp") + "/ds4_mini_engine.gguf";
        mini::Cfg mc;
        std::vector<std::vector<uint8_t>> keep;
        const std::string we = mini::write_gguf(mc, base, keep);
        ok("wrote the miniature fixture", we.empty(), we);
        keep.clear();
        const std::string pe = write_tokenizer_patched_gguf(base, path);
        ok("patched in a working 512-token tokenizer", pe.empty(), pe);
        std::remove(base.c_str());

        if (we.empty() && pe.empty()) {
            // Sanity on the patch itself BEFORE the engine sees it: the reader
            // must find the shadowing vocab, and the config's vocab must still be
            // the fixture's 512 (or DeepSeek4Model::load would reject token_embd).
            GgufReader g;
            const std::string oe = g.open(path);
            ok("patched fixture still parses", oe.empty(), oe);
            DeepSeek4Config pcfg;
            const std::string ce = oe.empty() ? read_deepseek4_config(g, pcfg) : oe;
            ok("read_deepseek4_config on the patched fixture", ce.empty(), ce);
            ok("vocab still 512 (tensors unchanged)", pcfg.vocab == mc.VOCAB,
               std::to_string(pcfg.vocab));
            ok("arch detects as deepseek4", detect_arch(g) == ModelArch::kDeepSeek4);
            g.close();

            EngineOptions eo;
            eo.max_ctx       = 448;      // fixture Ds4Options::max_context bound
            eo.prefill_chunk = 64;       // -> Ds4Options::max_seq
            eo.prompt_cache  = false;
            // Single card by default.  $DS4_TP_GPUS ("0,1") additionally exercises
            // Engine's DeepSeek4TpRuntime branch, exactly as §5/§7 use it.
            eo.n_gpus = std::max<uint32_t>(1u, uint32_t(tp_gpus.size()));

            std::string err;
            const double l0 = now_s();
            std::unique_ptr<Engine> eng = Engine::load(path, eo, err);
            const double load_s = now_s() - l0;
            ok(eo.n_gpus > 1 ? "Engine::load takes the deepseek4 TENSOR-PARALLEL branch"
                             : "Engine::load takes the deepseek4 single-card branch",
               eng != nullptr, err);
            if (eng) {
                ok("Engine::arch() == kDeepSeek4", eng->arch() == ModelArch::kDeepSeek4);
                ok("Engine::vocab() == the fixture's", eng->vocab() == mc.VOCAB,
                   std::to_string(eng->vocab()));
                std::printf("    load %.2f s over %u card(s)\n", load_s, eo.n_gpus);

                // Round-trip the prompt first: if encode/decode is not exact the
                // generated text says nothing about the engine.
                const std::string prompt =
                    "The quick brown fox jumps over the lazy dog. 0123456789";
                const auto pids = eng->tokenizer().encode(prompt, /*allow_special=*/true);
                const std::string back =
                    eng->tokenizer().decode(std::span<const int32_t>(pids), true,
                                            std::span<const int32_t>{});
                ok("prompt encodes to a non-empty id list", !pids.empty(),
                   std::to_string(pids.size()) + " ids");
                ok("prompt decodes back BYTE-IDENTICAL", back == prompt, printable(back));

                SamplingParams sp;
                sp.temperature = 0.0f;      // greedy: reproducible
                sp.max_tokens  = 24;
                sp.seed        = 1;
                std::string streamed;
                uint32_t chunks = 0;
                const GenerateResult res = eng->generate(prompt, sp,
                    [&](std::string_view t) { streamed.append(t); ++chunks; return true; });

                ok("Engine::generate produced tokens", res.completion_tokens > 0,
                   std::to_string(res.completion_tokens));
                ok("the streaming callback fired", chunks > 0, std::to_string(chunks) + " chunks");
                ok("streamed text == returned text", streamed == res.text);
                ok("generated text is NON-EMPTY (ids reached the tokenizer)", !res.text.empty(),
                   std::to_string(res.text.size()) + " bytes");
                std::printf("    prompt %u tok, %.1f ms prefill | %u new tok, %.1f ms decode\n",
                            res.prompt_tokens, res.prefill_ms,
                            res.completion_tokens, res.decode_ms);
                std::printf("    %sDETOKENIZED OUTPUT: \"%s\"%s\n",
                            G, printable(res.text).c_str(), Z);
                std::printf("    %s(the fixture's weights are pseudo-random -- this text is"
                            " MEANINGLESS by construction.\n"
                            "     It proves the id->text path runs, NOT that DeepSeek-V4"
                            " produces coherent English.)%s\n", Y, Z);

                // The chat path: Engine::chat's kDeepSeek4 branch must be the one
                // taken (it is checked before the template-family dispatch), so a
                // chat turn must render through build_deepseek4_prompt.  The
                // fixture's 512-token vocab has none of the DeepSeek sentinels, so
                // they encode to their literal characters -- which is enough to
                // prove the branch is REACHED and not enough to prove the encoding
                // is right (deepseek4_tokenizer_test owns that, against the real
                // vocab).
                const ChatTurn turns[] = {{"user", "hello"}};
                const GenerateResult cres =
                    eng->chat(std::span<const ChatTurn>(turns, 1), sp, {}, /*thinking=*/false);
                ok("Engine::chat reaches the deepseek4 branch (no unsupported-role error)",
                   cres.finish_reason.rfind("error:", 0) != 0, cres.finish_reason);
                ok("Engine::chat produced tokens", cres.completion_tokens > 0,
                   std::to_string(cres.completion_tokens));
                std::printf("    %schat DETOKENIZED OUTPUT: \"%s\"%s\n",
                            G, printable(cres.text).c_str(), Z);

                const ChatTurn bad[] = {{"developer", "x"}};
                const GenerateResult bres =
                    eng->chat(std::span<const ChatTurn>(bad, 1), sp, {}, false);
                ok("Engine::chat refuses an unsupported role by name",
                   bres.finish_reason.find("is not supported on the deepseek4 path")
                       != std::string::npos, bres.finish_reason);
            }
        }
        std::remove(path.c_str());
    }

    std::printf("\n%s%d failure(s)%s\n", g_fail ? R : G, g_fail, Z);
    return g_fail ? 1 : 0;
}
