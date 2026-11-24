// src/engine/xing4_engine.cpp -- Xing4.0-29B-A4B (GGUF arch `xing4_0`) in the engine: load, the chat template, the request
// lanes, parse. The runtime is Xing4Model (src/model/xing4.cpp) on the two cards; the tokenizer is the GGUF's; the template is
// the release's chat_template.jinja rendered natively (xing4_render_prompt, byte-exact against transformers on the cases in
// scripts/xing4_oracle/template_cases.json). Tool calls are GLM's with <param_key>/<param_value> for <arg_key>/<arg_value>:
// the reply is translated to GLM's spelling token by token and parsed by GLM's parser (the Hy4-flash route).
//
// Lanes: Xing4LanesModel puts the model under the shared lanes server (ie/lanes_serve.hpp). Every lane is a sequence in the
// model's paged cache (Xing4Model::forward_rows takes rows of any lanes), so the lane pipe is one worker thread that runs
// every queued step -- decode rows and prompt pieces of any lanes -- as ONE forward, up to the model's rows a call. A lane
// keeps its sequence after a request; the next request on it reuses the common prefix (own_match).
#include "ie/engine.hpp"

#include "ie/deepseek41_generate.hpp"   // Ds41Generator::sample (the host sampler)
#include "ie/glm5_server.hpp"           // parse_glm5_completion
#include "ie/lanes_auto.hpp"
#include "ie/lanes_serve.hpp"
#include "ie/allocator.hpp"
#include "ie/xing4_engine.hpp"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <thread>

namespace ie {

namespace {
using ojson = nlohmann::ordered_json;

// (the model's tag, the stream's)
constexpr std::pair<const char*, const char*> kTags[] = {
    {"<param_key>", "<arg_key>"}, {"</param_key>", "</arg_key>"}, {"<param_value>", "<arg_value>"}, {"</param_value>", "</arg_value>"},
};

// Jinja's tojson(ensure_ascii=False): Python json.dumps with ", " and ": "
std::string py_dumps(const ojson& j) {
    if (j.is_object()) {
        std::string o = "{"; bool first = true;
        for (auto it = j.begin(); it != j.end(); ++it) { if (!first) o += ", "; first = false; o += ojson(it.key()).dump(-1, ' ', false) + ": " + py_dumps(it.value()); }
        return o + "}";
    }
    if (j.is_array()) {
        std::string o = "["; bool first = true;
        for (const auto& v : j) { if (!first) o += ", "; first = false; o += py_dumps(v); }
        return o + "]";
    }
    return j.dump(-1, ' ', false, ojson::error_handler_t::replace);
}

std::string strip(std::string_view s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(uint8_t(s[a]))) ++a;
    while (b > a && std::isspace(uint8_t(s[b - 1]))) --b;
    return std::string(s.substr(a, b - a));
}
std::string lstrip_nl(std::string_view s) { size_t a = 0; while (a < s.size() && s[a] == '\n') ++a; return std::string(s.substr(a)); }
std::string rstrip_nl(std::string_view s) { size_t b = s.size(); while (b > 0 && s[b - 1] == '\n') --b; return std::string(s.substr(0, b)); }
}  // namespace

std::string xing4_render_prompt(std::span<const ChatTurn> turns, std::string_view tools_json, bool enable_thinking, std::string& err,
                                std::vector<std::pair<size_t, size_t>>* content) {
    std::string o;
    auto text = [&](std::string_view s) { if (content && !s.empty()) content->emplace_back(o.size(), s.size()); o += s; };
    size_t first = 0;
    o += "<_system>";
    if (!turns.empty() && turns[0].role == "system") { text(turns[0].content); first = 1; }
    if (!tools_json.empty()) {
        ojson tools = ojson::parse(tools_json, nullptr, false);
        if (tools.is_discarded() || !tools.is_array()) { err = "tools: not a JSON array"; return {}; }
        if (!tools.empty()) {
            o += "\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures "
                 "within <tools></tools> XML tags:\n<tools>\n";
            for (const auto& t : tools) { text(py_dumps(t)); o += "\n"; }
            o += "</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n"
                 "<tool_call>{function-name}<param_key>{param-key-1}</param_key><param_value>{param-value-1}</param_value><param_key>"
                 "{param-key-2}</param_key><param_value>{param-value-2}</param_value>...</tool_call>";
        }
    }
    long last_user = -1;
    for (size_t i = first; i < turns.size(); ++i) if (turns[i].role == "user") last_user = long(i - first);
    for (size_t i = first; i < turns.size(); ++i) {
        const ChatTurn& t = turns[i];
        const long idx = long(i - first);
        if (t.role == "user") { o += "<_user>"; text(t.content); }
        else if (t.role == "assistant") {
            o += "<_bot>";
            std::string reasoning, body = t.content_without_tool_calls ? *t.content_without_tool_calls : t.content;
            if (t.reasoning_content) reasoning = *t.reasoning_content;
            else if (const size_t e = body.find("</think>"); e != std::string::npos) {
                std::string before = rstrip_nl(body.substr(0, e));
                if (const size_t s = before.rfind("<think>"); s != std::string::npos) before = before.substr(s + 7);
                reasoning = lstrip_nl(before);
                body = lstrip_nl(body.substr(body.rfind("</think>") + 8));
            }
            if (idx < last_user || !enable_thinking) o += "</think>";
            else { o += "<think>\n"; text(strip(reasoning)); o += "\n</think>"; }
            const std::string bs = strip(body);
            if (!bs.empty()) text(bs);
            if (!t.tool_calls_json.empty() && t.tool_calls_json != "[]") {
                ojson calls = ojson::parse(t.tool_calls_json, nullptr, false);
                if (calls.is_discarded() || !calls.is_array()) { err = "assistant tool_calls: not a JSON array"; return {}; }
                for (const auto& c0 : calls) {
                    const ojson& c = c0.contains("function") ? c0["function"] : c0;
                    o += "<tool_call>"; text(c.value("name", ""));
                    ojson args = c.contains("arguments") ? c["arguments"] : ojson::object();
                    if (args.is_string()) { ojson p = ojson::parse(args.get<std::string>(), nullptr, false); args = p.is_discarded() ? ojson::object() : p; }
                    if (args.is_object())
                        for (auto it = args.begin(); it != args.end(); ++it) {
                            o += "<param_key>"; text(it.key()); o += "</param_key><param_value>";
                            text(it.value().is_string() ? it.value().get<std::string>() : py_dumps(it.value()));
                            o += "</param_value>";
                        }
                    o += "</tool_call>";
                }
            }
            o += "<_end>\n";
        } else if (t.role == "tool") {
            if (idx > 0 && turns[i - 1].role != "tool") o += "<_observation>";
            o += "<tool_response>"; text(t.content); o += "</tool_response>";
        } else if (t.role == "system") { o += "<_system>"; text(t.content); }
    }
    o += "<_bot>";
    o += enable_thinking ? "<think>\n" : "</think>";
    return o;
}

std::vector<int32_t> xing4_encode(const Tokenizer& tok, std::string_view prompt, std::span<const std::pair<size_t, size_t>> content) {
    std::vector<int32_t> ids = tok.encode(prompt, /*allow_special=*/true, content);
    if (!ids.empty() && tok.add_bos_token() && ids[0] == tok.bos_token_id()) ids.erase(ids.begin());   // the template writes no BOS
    // "<_observation>" is a control token of the vocabulary that transformers' tokenizer never produces from text: it reads
    // the tag as "<", "_", "observation", ">" (template_cases.json "tools"), and the model saw those pieces
    const int32_t obs = tok.find_token("<_observation>");
    const int32_t p[4] = {tok.find_token("<"), tok.find_token("_"), tok.find_token("observation"), tok.find_token(">")};
    if (obs >= 0 && p[0] >= 0 && p[1] >= 0 && p[2] >= 0 && p[3] >= 0 && std::find(ids.begin(), ids.end(), obs) != ids.end()) {
        std::vector<int32_t> o;
        o.reserve(ids.size() + 8);
        for (int32_t id : ids) { if (id == obs) o.insert(o.end(), p, p + 4); else o.push_back(id); }
        ids.swap(o);
    }
    return ids;
}

std::string xing4_translate_reply(std::string_view raw) {
    std::string s(raw);
    for (const auto& [from, to] : kTags) {
        const std::string_view f(from), t(to);
        for (size_t at = s.find(f); at != std::string::npos; at = s.find(f, at + t.size())) s.replace(at, f.size(), t);
    }
    return s;
}

// ------------------------------------------------------------------------------------------------------------------ lanes
class Xing4LanesModel final : public LanesModel {
public:
    Xing4LanesModel(Xing4Model& m, const Tokenizer& tok, std::vector<int32_t> eos, uint32_t lanes, uint32_t cap, uint32_t chunk)
        : m_(m), tok_(tok), eos_(std::move(eos)), n_(lanes), cap_(cap), chunk_(chunk), V_(m.config().vocab),
          hist_(lanes), logits_(lanes, std::vector<float>(m.config().vocab, 0.f)), end_(lanes, 0) {
        for (const auto& [from, to] : kTags) {
            const auto e = tok.encode(from, true);
            if (e.size() == 1) tag_text_.emplace_back(e[0], to);
        }
    }
    ~Xing4LanesModel() override { (void)pipe_stop(); }
    bool tags_single() const { return tag_text_.size() == std::size(kTags); }

    const char* tag() const override { return "xing4 lanes"; }
    uint32_t n_lanes() const override { return n_; }
    uint32_t vocab() const override { return V_; }
    uint32_t lane_cap(uint32_t) const override { return cap_; }
    uint32_t own_match(uint32_t lane, std::span<const int32_t> ids) const override {
        const auto& h = hist_[lane];
        size_t L = 0;
        while (L < h.size() && L + 1 < ids.size() && h[L] == ids[L]) ++L;   // at least one row to read
        return uint32_t(L);
    }
    bool occupied(uint32_t lane) const override { return !hist_[lane].empty(); }
    uint32_t last_end(uint32_t lane) const override { return end_[lane]; }
    uint32_t min_match_tokens() const override { return 64; }
    std::string prefix_prepare(uint32_t lane, const LanesRequest& rq, uint32_t& reused, std::string& source) override {
        reused = own_match(lane, *rq.ids);
        std::lock_guard<std::mutex> lk(fw_mu_);
        // the longest opening another lane holds (Dream's system prompt and tools, an earlier turn of this conversation): its
        // full cache pages are shared into this lane -- no copy, no re-read (IE_XG4_SHARE=0: own lane only)
        static const bool share = !(std::getenv("IE_XG4_SHARE") && *std::getenv("IE_XG4_SHARE") == '0');
        uint32_t best = reused, donor = lane;
        if (share)
            for (uint32_t d = 0; d < n_; ++d)
                if (d != lane) { const uint32_t m = own_match(d, *rq.ids); if (m > best) { best = m; donor = d; } }
        if (donor != lane && best >= reused + 64) {
            if (auto e = m_.lane_share(lane, donor, best); !e.empty()) return e;
            reused = m_.lane_len(lane);
            hist_[lane].assign(rq.ids->begin(), rq.ids->begin() + reused);
            ++shares_; shared_tokens_ += reused;
            source = "lane " + std::to_string(donor) + " (shared)";
            return {};
        }
        m_.lane_truncate(lane, reused);
        hist_[lane].resize(reused);
        source = reused ? "lane" : "";
        return {};
    }
    LanesPlan plan(const LanesRequest& rq, uint32_t reused) override {
        LanesPlan p;
        const uint32_t T = uint32_t(rq.ids->size());
        for (uint32_t pos = reused; pos < T; pos += chunk_) p.chunks.emplace_back(pos, std::min(chunk_, T - pos));
        p.Tp = T;
        return p;
    }
    std::string prefill_serial(uint32_t lane, const LanesRequest& rq, std::span<const LanesChunk> chunks, const std::function<bool()>& stop,
                               size_t& done) override {
        done = 0;
        for (size_t i = 0; i < chunks.size(); ++i) {
            if (i > 0 && stop && stop()) break;
            if (auto e = run({Step{lane, std::vector<int32_t>(rq.ids->begin() + chunks[i].first, rq.ids->begin() + chunks[i].first + chunks[i].second),
                                   chunks[i].first}}); !e.empty()) return e;
            ++done;
        }
        return {};
    }
    std::string prompt_end(uint32_t lane, const LanesRequest& rq, uint32_t, uint32_t, bool) override {
        end_[lane] = uint32_t(rq.ids->size());
        return {};
    }
    std::string reset_lane(uint32_t lane) override {
        std::lock_guard<std::mutex> lk(fw_mu_);
        m_.lane_truncate(lane, 0);
        hist_[lane].clear(); end_[lane] = 0;
        return {};
    }
    int32_t sample(uint32_t lane, bool, const LanesSampling& sp, std::span<const int32_t> window, uint64_t seed, std::string& err) override {
        std::vector<float>& lg = logits_[lane];
        for (float v : lg) if (!std::isfinite(v)) { err = "non-finite logits"; return -1; }
        Ds41SampleParams p; p.temperature = sp.temperature; p.top_k = sp.top_k; p.top_p = sp.top_p; p.min_p = sp.min_p;
        p.repeat_penalty = sp.repeat_penalty; p.repeat_window = sp.repeat_window; p.seed = seed;
        uint64_t rng = seed;
        std::vector<float> work(lg);
        if (sp.filter) sp.filter->apply(work);   // the tool-call guard
        const std::vector<int32_t> recent(window.begin(), window.end());
        return Ds41Generator::sample(work, recent, p, rng);
    }
    void keep_logits(uint32_t) override {}   // every step keeps its last row's logits
    bool is_stop(int32_t id) const override { return std::find(eos_.begin(), eos_.end(), id) != eos_.end(); }
    std::string detok(std::span<const int32_t> out) const override {
        std::string s;
        size_t a = 0;
        auto flush = [&](size_t b) { if (b > a) s += tok_.decode(out.subspan(a, b - a), false); };
        for (size_t i = 0; i < out.size(); ++i) {
            const auto it = std::find_if(tag_text_.begin(), tag_text_.end(), [&](const auto& e) { return e.first == out[i]; });
            if (it == tag_text_.end()) continue;
            flush(i); s += it->second; a = i + 1;
        }
        flush(out.size());
        return s;
    }

    // ---- the pipe: one worker; every queued step of every lane in one forward ----
    std::string pipe_start(DoneFn done) override { return start(std::move(done), nullptr); }
    bool rows() const override { return true; }
    std::string pipe_start_rows(DoneFn done, RowsDoneFn rows_done) override { return start(std::move(done), std::move(rows_done)); }
    std::string pipe_submit(uint32_t lane, const int32_t* ids, uint32_t T, uint32_t pos0) override {
        std::lock_guard<std::mutex> lk(mu_);
        if (!err_.empty()) return err_;
        q_.push_back(Step{lane, std::vector<int32_t>(ids, ids + T), pos0});
        cv_.notify_all();
        return {};
    }
    std::string pipe_pause() override {
        std::unique_lock<std::mutex> lk(mu_);
        paused_ = true;
        cv_.wait(lk, [&] { return !busy_; });
        return err_;
    }
    std::string pipe_resume() override {
        std::lock_guard<std::mutex> lk(mu_);
        paused_ = false;
        cv_.notify_all();
        return err_;
    }
    bool pipe_paused() const override { std::lock_guard<std::mutex> lk(mu_); return paused_; }
    std::string pipe_stop() override {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
            cv_.notify_all();
        }
        if (th_.joinable()) th_.join();
        std::lock_guard<std::mutex> lk(mu_);
        q_.clear(); stop_ = false; paused_ = false;
        return {};
    }
    std::string pipe_error() const override { std::lock_guard<std::mutex> lk(mu_); return err_; }
    uint32_t pipe_max_rows() const override { return chunk_; }
    uint32_t abort() override {
        std::lock_guard<std::mutex> lk(mu_);
        q_.clear();
        return busy_ ? 1u : 0u;
    }

private:
    struct Step { uint32_t lane; std::vector<int32_t> ids; uint32_t pos0; };

    // the steps as one forward (fw_mu_ held inside): each step's last row's logits into its lane's buffer
    std::string run(const std::vector<Step>& steps) {
        std::lock_guard<std::mutex> lk(fw_mu_);
        std::vector<Xing4Row> rows;
        std::vector<uint32_t> lr;
        for (const Step& s : steps) {
            if (s.pos0 < m_.lane_len(s.lane)) { m_.lane_truncate(s.lane, s.pos0); hist_[s.lane].resize(s.pos0); }
            if (s.pos0 != m_.lane_len(s.lane))
                return "lane " + std::to_string(s.lane) + ": step at " + std::to_string(s.pos0) + " but the lane holds " + std::to_string(m_.lane_len(s.lane));
            for (int32_t id : s.ids) rows.push_back({id, s.lane});
            lr.push_back(uint32_t(rows.size() - 1));
        }
        out_.resize(size_t(lr.size()) * V_);
        if (auto e = m_.forward_rows(rows.data(), uint32_t(rows.size()), out_.data(), lr.data(), uint32_t(lr.size())); !e.empty()) return e;
        for (size_t i = 0; i < steps.size(); ++i) {
            std::copy(out_.begin() + long(i * V_), out_.begin() + long((i + 1) * V_), logits_[steps[i].lane].begin());
            hist_[steps[i].lane].insert(hist_[steps[i].lane].end(), steps[i].ids.begin(), steps[i].ids.end());
        }
        return {};
    }
    std::string start(DoneFn done, RowsDoneFn rows_done) {
        std::lock_guard<std::mutex> lk(mu_);
        if (th_.joinable()) return "the pipe is running";
        done_ = std::move(done); rows_done_ = std::move(rows_done);
        err_.clear(); paused_ = false; stop_ = false;
        th_ = std::thread([this] { loop(); });
        return {};
    }
    void loop() {
        for (;;) {
            std::vector<Step> batch;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return stop_ || (!paused_ && !q_.empty() && err_.empty()); });
                if (stop_) return;
                uint32_t rows = 0;
                while (!q_.empty() && (batch.empty() || rows + q_.front().ids.size() <= chunk_)) {
                    rows += uint32_t(q_.front().ids.size());
                    batch.push_back(std::move(q_.front()));
                    q_.pop_front();
                }
                busy_ = true;
            }
            std::string e;
            try { e = run(batch); }
            catch (const std::exception& x) { e = std::string("threw: ") + x.what(); }
            {
                std::lock_guard<std::mutex> lk(mu_);
                busy_ = false;
                if (!e.empty()) err_ = e;
                cv_.notify_all();
            }
            if (!e.empty()) { std::fprintf(stderr, "[xing4 lanes] pipe step failed: %s\n", e.c_str()); continue; }
            std::vector<uint32_t> lanes;
            for (const Step& s : batch) lanes.push_back(s.lane);
            if (lanes.size() >= 2 && rows_done_) rows_done_(lanes);
            else for (uint32_t l : lanes) done_(l);
        }
    }

    Xing4Model& m_;
    const Tokenizer& tok_;
    std::vector<int32_t> eos_;
    std::vector<std::pair<int32_t, std::string>> tag_text_;
    uint32_t n_, cap_, chunk_, V_;
    std::vector<std::vector<int32_t>> hist_;   // the tokens each lane's cache holds
    std::vector<std::vector<float>> logits_;   // each lane's latest row
    std::vector<uint32_t> end_;
    std::vector<float> out_;
    uint64_t shares_ = 0, shared_tokens_ = 0;   // prompts that took another lane's opening, and its tokens
    std::mutex fw_mu_;                         // one forward at a time
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Step> q_;
    bool paused_ = false, stop_ = false, busy_ = false;
    std::string err_;
    std::thread th_;
    DoneFn done_;
    RowsDoneFn rows_done_;
};

Xing4Bundle::~Xing4Bundle() {
    serve.reset();
    lanes.reset();
}

// ------------------------------------------------------------------------------------------------------------- engine
bool Engine::xing4_gguf(const std::string& path) {
    if (path.size() < 5 || path.compare(path.size() - 5, 5, ".gguf") != 0) return false;
    GgufReader g;
    if (!g.open(path).empty()) return false;
    const KvRef* a = g.find_kv("general.architecture");
    return a && a->as_string() == "xing4_0";
}

std::string Engine::xing4_load(const std::string& path) {
    auto b = std::make_unique<Xing4Bundle>();
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<sycl::device> cards;
    for (const auto& pl : sycl::platform::get_platforms()) {
        if (pl.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : pl.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) cards.push_back(d);
    }
    if (cards.empty()) return "xing4: no Arc GPU";
    const uint32_t n = std::min<uint32_t>(opts_.n_gpus >= 1 ? std::min<uint32_t>(opts_.n_gpus, 2) : 2, uint32_t(cards.size()));
    opts_.n_gpus = n;
    // two cards in ONE context with peer access (IE_XG4_P2P=0: a context each): the streams move card to card as a push
    std::unique_ptr<sycl::context> p2p_ctx;
    if (n == 2 && !(std::getenv("IE_XG4_P2P") && *std::getenv("IE_XG4_P2P") == '0')) {
        std::vector<sycl::device> pg;
        if (gpu_p2p_shared_context("Arc", pg, p2p_ctx) && pg.size() >= 2) cards = {pg[0], pg[1]};
        else p2p_ctx.reset();
    }
    std::vector<sycl::queue*> qs;
    for (uint32_t i = 0; i < n; ++i) {
        b->queues.push_back(std::make_unique<sycl::queue>(p2p_ctx ? *p2p_ctx : sycl::context(cards[i]), cards[i], sycl::property::queue::in_order{}));
        qs.push_back(b->queues.back().get());
    }
    uint32_t lanes = opts_.parallel;
    if (lanes == kLanesAuto) {
        lanes = kLanesAutoMax;
        std::fprintf(stderr, "%s\n", lanes_auto_line("xing4", lanes, opts_.max_ctx, "lanes share one paged cache; a lane takes pages as it grows").c_str());
        opts_.parallel = lanes;
    }
    lanes = std::max<uint32_t>(1, lanes);
    Xing4Options o;
    o.max_ctx = std::max<uint32_t>(1024, opts_.max_ctx);
    o.max_lanes = lanes;
    o.max_tokens = opts_.prefill_chunk_set ? std::max<uint32_t>(16, std::min<uint32_t>(opts_.prefill_chunk, 2048)) : 512u;
    // the paged cache: what the cards hold after the weights (half the file a card) and a 2.5 GiB margin, 1152 bytes a key
    // a layer, at most every lane at its full context. IE_XG4_KV_TOKENS sets it.
    {
        uint64_t file = 0;
        if (FILE* f = std::fopen(path.c_str(), "rb")) { std::fseek(f, 0, SEEK_END); file = uint64_t(std::ftell(f)); std::fclose(f); }
        uint64_t fr = UINT64_MAX;
        for (uint32_t i = 0; i < n; ++i)
            if (cards[i].has(sycl::aspect::ext_intel_free_memory)) fr = std::min<uint64_t>(fr, cards[i].get_info<sycl::ext::intel::info::device::free_memory>());
        if (fr == UINT64_MAX) fr = 30ull << 30;
        const uint64_t per_card = file / n + (5ull << 29), layers_card = (40 + n - 1) / n;
        const uint64_t room = fr > per_card ? fr - per_card : 0;
        uint64_t keys = room / (layers_card * 1152ull);
        keys = std::min<uint64_t>(keys, uint64_t(lanes) * o.max_ctx);
        if (const char* v = std::getenv("IE_XG4_KV_TOKENS"); v && *v) keys = std::strtoull(v, nullptr, 10);
        if (keys < o.max_ctx) return "xing4: the cards hold a cache of " + std::to_string(keys) + " keys, below one lane at ctx " + std::to_string(o.max_ctx);
        o.kv_tokens = uint32_t(std::min<uint64_t>(keys, 0xFFFFFFC0ull));
    }
    if (auto e = b->model.load(path, qs, o); !e.empty()) return "xing4: " + e;
    if (auto e = b->tok.load_from_gguf(b->model.gguf()); !e.empty()) return "xing4 tokenizer: " + e;
    auto add = [&](int32_t id) { if (id >= 0 && std::find(b->eos.begin(), b->eos.end(), id) == b->eos.end()) b->eos.push_back(id); };
    add(b->tok.eos_token_id());
    if (const auto e = b->tok.encode("<_end>", true); e.size() == 1) add(e[0]);
    if (const auto e = b->tok.encode("</think>", true); e.size() == 1) { b->think_close = e[0]; b->tok.show_special(e[0]); }
    if (const auto e = b->tok.encode("<think>", true); e.size() == 1) b->tok.show_special(e[0]);
    for (const char* tg : {"<tool_call>", "</tool_call>", "<param_key>", "</param_key>", "<param_value>", "</param_value>"})
        if (const auto e = b->tok.encode(tg, true); e.size() == 1) b->tok.show_special(e[0]);
    b->lanes = std::make_unique<Xing4LanesModel>(b->model, b->tok, b->eos, lanes, o.max_ctx - 8, o.max_tokens);
    if (!b->lanes->tags_single()) std::fprintf(stderr, "[xing4] the tool-call tags are not single tokens: the stream carries them as the model writes them; the reply is translated when it ends\n");
    LanesServe::Options so;
    b->serve = std::make_unique<LanesServe>(*b->lanes, so);
    prompt_cache_on_ = opts_.prompt_cache;
    std::fprintf(stderr, "[xing4] ready in %.1f s: %u card(s), %u request lanes at ctx up to %u each, a cache of %u keys shared by the lanes, prompt chunk %u, eos",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), n, lanes, o.max_ctx, b->model.free_keys(), o.max_tokens);
    for (int32_t e : b->eos) std::fprintf(stderr, " %d", e);
    std::fprintf(stderr, "\n");
    arch_ = ModelArch::kXing4;
    xg4_ = std::move(b);
    return {};
}

namespace {
GenerateResult xing4_run(Xing4Bundle& b, const std::vector<int32_t>& ids, const SamplingParams& sp, const TokenCallback& on_token,
                         std::string_view tools_json = {}, bool thinking = false) {
    GenerateResult res;
    res.prompt_tokens = uint32_t(ids.size());
    if (ids.empty()) { res.finish_reason = "error: empty prompt"; return res; }
    LanesRequest rq;
    rq.ids = &ids;
    rq.sp.temperature = sp.temperature; rq.sp.top_k = sp.top_k; rq.sp.top_p = sp.top_p; rq.sp.min_p = sp.min_p;
    rq.sp.presence_penalty = sp.presence_penalty; rq.sp.frequency_penalty = sp.frequency_penalty;
    rq.sp.repeat_penalty = sp.repeat_penalty; rq.sp.repeat_window = sp.repeat_window;
    rq.sp.max_tokens = sp.max_tokens; rq.sp.ignore_eos = sp.ignore_eos;
    rq.rng = sp.seed ? sp.seed : uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
    rq.think_close = b.think_close;
    // every tool call valid: the request's tools guard the sampler (IE_TOOL_GUARD=0 turns it off)
    static const bool guard_on = !(std::getenv("IE_TOOL_GUARD") && *std::getenv("IE_TOOL_GUARD") == '0');
    if (guard_on && sp.tool_guard && !tools_json.empty()) {
        std::string gerr;
        rq.guard = ToolGuard::make(tools_json, ToolGuard::xing4_tags(), gerr);
        rq.guard_think = thinking;
        if (!gerr.empty()) std::fprintf(stderr, "[xing4] tool-call guard off for this request: %s\n", gerr.c_str());
        if (rq.guard && !sp.tool_choice.empty()) {   // tool_choice none / required / a function
            rq.guard = rq.guard->with_choice(sp.tool_choice, gerr);
            if (!rq.guard) { res.finish_reason = "error: " + gerr; return res; }
        }
    }
    if (rq.sp.max_tokens == 0) { res.finish_reason = "length"; return res; }
    const LanesResult lr = b.serve->run(rq, on_token);
    res.text = lr.text;
    res.answer_start = lr.answer_start;
    res.finish_reason = lr.finish_reason;
    res.cached_tokens = lr.cached_tokens;
    res.lane = lr.lane; res.lane_ctx = lr.lane_ctx;
    res.completion_tokens = lr.completion_tokens;
    res.prefill_ms = lr.prefill_ms;
    res.decode_ms = lr.decode_ms;
    res.cache_source = lr.cache_source;
    return res;
}
}  // namespace

GenerateResult Engine::xing4_generate(const std::string& prompt, const SamplingParams& sp, const TokenCallback& on_token) {
    const auto ids = xg4_->tok.encode(prompt, /*allow_special=*/true);
    return xing4_run(*xg4_, ids, sp, on_token);
}

GenerateResult Engine::xing4_chat(std::span<const ChatTurn> turns, const SamplingParams& sp, const TokenCallback& on_token,
                                  bool enable_thinking, std::string_view tools_json) {
    GenerateResult r;
    std::string err;
    std::vector<std::pair<size_t, size_t>> content;
    const auto tr0 = std::chrono::steady_clock::now();
    const std::string prompt = xing4_render_prompt(turns, tools_json, enable_thinking, err, &content);
    const auto tr1 = std::chrono::steady_clock::now();
    if (!err.empty()) { r.finish_reason = "error: " + err; return r; }
    for (const auto& t : turns) if (!t.images.empty()) { r.finish_reason = "error: xing4 reads no images"; return r; }
    const auto ids = xing4_encode(xg4_->tok, prompt, content);
    {
        const auto tr2 = std::chrono::steady_clock::now();
        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        std::fprintf(stderr, "[xing4] request: render %.1f ms (%zu chars), tokenize %.1f ms (%zu tokens)\n", ms(tr0, tr1), prompt.size(), ms(tr1, tr2), ids.size());
    }
    r = xing4_run(*xg4_, ids, sp, on_token, tools_json, enable_thinking);
    if (r.finish_reason == "stop" || r.finish_reason == "length") {
        auto pc = parse_glm5_completion(xing4_translate_reply(r.text), enable_thinking, tools_json);
        r.reasoning_content = std::move(pc.reasoning);
        r.text = std::move(pc.content);
        if (!r.text.empty() && r.text[0] == ' ') r.text.erase(0, 1);   // (the first piece's word-start space)
        r.tool_calls_json = pc.tool_calls.empty() ? "[]" : std::move(pc.tool_calls);
        if (r.tool_calls_json != "[]" && r.finish_reason == "stop") r.finish_reason = "tool_calls";
    }
    return r;
}

}  // namespace ie
