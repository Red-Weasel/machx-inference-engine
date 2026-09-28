// include/ie/mimo26_engine.hpp — the MiMo-V2.6 bundle behind the Engine (P3b, docs/mimo26/00_PORT_PLAN.md): the
// model, the tokenizer.json tokenizer, one in-order queue per Arc card, the two-card forward, and the live
// conversation (the ids whose K/V the caches hold). Built and used in src/engine/mimo26_engine.cpp; complete here so
// Engine's destructor can destroy the unique_ptr. One request at a time at --parallel 1 (the path the engine always had);
// --parallel N > 1 serves N requests at once on N lanes (P4 B4, Mimo26Serve below, docs/mimo26/P4_B4_SERVE.md).
#pragma once
#include "ie/allocator.hpp"
#include "ie/deepseek41_generate.hpp"   // Ds41SampleParams: a lane's sampler settings
#include "ie/mimo26.hpp"
#include "ie/mimo26_dflash.hpp"
#include "ie/mimo26_forward.hpp"
#include "ie/mimo26_host_rules.hpp"   // mimo26_lanes_draft (P4 B16)
#include "ie/mimo26_prefix_cache.hpp"
#include "ie/mimo26_vision.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace ie {

// P4 B4 (docs/mimo26/P4_B4_SERVE.md): --parallel N > 1. Every request owns one LANE (Mimo26Forward's caches and positions,
// the drafter's context ring, the prefix cache's per-lane bookkeeping) from admission until it has read its outcome; the
// idle lanes keep their conversations for the prefix step of a later prompt. Decode steps and prefill chunks run through the
// lane pipe (Mimo26Forward::pipe_*, B2/B3): the pipe's done callback -- on the last card's stage thread -- samples a lane's
// row(s) on the host, commits the ids to the lane's outbox and submits the lane's next step; the request's own thread only
// takes ids from the outbox, decodes them to text and runs the server's callback. The prefix step and the prompt-end
// snapshot need the serial forward API (select_lane, state_spans, set_state, rewind), so a request takes the SERIAL TURN for
// them: with `pause` set the callbacks park their lanes instead of resubmitting, pipe_pause() drains the steps in flight (the
// stage threads stay, with their oneDNN contexts), the turn holder works alone on the cards, and releasing the turn resumes
// the pipe with every parked lane's step. One turn at a time; `mu` guards everything below.
struct Mimo26Serve {
    struct Tok { int32_t id = 0; bool has_H = false; double H = 0, margin = 0; };   // a committed id and the sampler's stats (ie_vitals)
    struct Lane {
        enum class Phase : uint8_t { kIdle, kPrefill, kPromptReady, kDecode, kDone };
        bool     busy = false;                    // owned by a request
        uint32_t cap = 0;                         // context capacity: lane 0 --ctx, the others --slot-ctx
        Phase    phase = Phase::kIdle;
        std::vector<int32_t> prompt, live;        // the request's ids; the ids at the lane's positions [0, n_pos), kept while idle
        uint32_t prefill_at = 0;                  // the next prompt position to run
        uint32_t max_new = 0, n_new = 0;
        int32_t  next = -1;                       // the sampled id whose forward is due
        Ds41SampleParams sp; bool ignore_eos = false, vitals = false;   // vitals: the request asked for ie_vitals (stats per sample)
        uint64_t rng = 0; std::vector<int32_t> recent;   // the sampler's state and repetition window (prompt tail + output)
        std::vector<Tok> outbox;                  // committed ids the request thread has not taken
        std::vector<std::pair<uint32_t, uint32_t>> drafts;   // draft passes (offered, accepted) it has not taken
        std::string finish;                       // "" while running; "stop" / "length" / "abort" / "error: ..."
        bool want_stop = false;                   // the request thread's cancel: honoured at the lane's next completion
        bool lost = false;                        // the lane's device state must be cleared (a target fault, a pipe error)
        bool parked = false;                      // a step is due and not submitted (the pipe was paused, or the lane is new)
        std::vector<float> first_logits, scratch; // the prompt's last row (kPromptReady); the sampler's copy of a row
        std::vector<int32_t> rows; uint32_t pend = 0;   // the step in flight: its ids (a decode step), or its prefill rows
        bool df_on = false; uint32_t df_pass = 0, df_acc = 0; std::string df_reason;   // the drafter follows the lane; off after a fault
        uint64_t tick = 0;                        // the lane's last release (larger = more recent): the LRU idle lane is taken first
        std::chrono::steady_clock::time_point t_sub{};
        uint32_t steps = 0, chunks = 0;
        double   cb_ms = 0, samp_ms = 0, dfctx_ms = 0;   // (diagnostics) the done callback's time for this request, of it the verify loop / the drafter's context
        double   draft_ms = 0;                    // (diagnostic) the drafter's draft passes for this request (in the callback, or at a turn's release)
        // P4 B5: the lane's drafts kept, as a per-draft acceptance estimate (mimo26_draft_acc): decayed counts of kept drafts and of
        // verify steps that rejected one; and the verify steps it took with drafts while other lanes decoded
        double   acc_kept = 4.0, acc_miss = 1.0;
        uint32_t shared_passes = 0;
        // P4 B16: a decode step's drafter context feed was skipped (adaptive drafting: n_dec >= draft_max_lanes), so the drafter's
        // context stops short of the lane's position -- the lane drafts nothing until a later step feeds it again (the gap rule)
        bool     df_gap = false;
    };
    std::vector<Lane>       lanes;                // empty at --parallel 1 (the serial path)
    std::mutex              mu;
    std::condition_variable cv;
    bool     piping = false, pause = false, turn_busy = false, stopping = false;   // piping: the pipe takes steps (started, not paused)
    uint32_t turn_waiters = 0;
    uint64_t tick = 0;
    // IE_MIMO26_MIX_CHUNK: a prefill chunk's rows while another lane is busy. Default 2,048 = the forward's full chunk (no cap):
    // measured against 512 (docs/mimo26/P4_B4_SERVE.md, criterion d), a MiMo chunk's cost is mostly per chunk (a 91-row chunk
    // 1.5 s, 512 rows 2.4-4.7 s, 1,622 rows 3.9 s), so 512 doubled the arrival's TTFT and stalled the decoding lanes longer
    uint32_t mix_chunk = 2048;
    // /health: the decode steps (submit to callback), an exponential average, and their rows; the ids committed (every lane)
    double   step_ms = 0, rows_per_step = 0;
    uint64_t steps = 0, tokens = 0;
    uint64_t turns = 0; double paused_ms = 0;     // serial turns taken, and the time the pipe spent paused for them (the turn's work and the resume; the drain before it is not counted)
    uint64_t runs = 0;                            // the pipe's starts and resumes (IE_MIMO26_STEP_TRACE: steps are timed from the latest)
    std::chrono::steady_clock::time_point turn_t0{}, run_t0{};
    // P4 B5 (docs/mimo26/P4_B5_DRAFT_BUDGET.md): the drafter with several lanes decoding. IE_MIMO26_DRAFT_BUDGET = rows per group
    // step (<= Mimo26Forward::kDecodeRows); 0 = the B4 rule (the drafter drafts only while one lane decodes). IE_MIMO26_DRAFT_WEIGHT=1:
    // the group's rows are split by each lane's acceptance, and rows a lane's drafter cut leaves go to the lanes after it in the
    // group; 0 = every lane the same share. IE_MIMO26_GROUP_LANES: the pipe's lanes per group (0 = its AUTO). The budget is also the
    // pipe's row cap for a group, so a budget below the lanes per group splits groups (1 = one plain lane per group: use 0 instead).
    uint32_t draft_budget = 0, group_lanes = 0;
    bool     draft_weight = false;
    uint32_t cb_used = 0;                         // (weighted) the rows the earlier lanes of the callback group being answered resubmitted
    // /health: draft passes (every lane, since load), their time, the drafts offered and kept, and the passes with other lanes decoding;
    // the decode steps that ran in a group of several lanes (of `steps`: whether the pipe's AUTO pairing engages)
    uint64_t df_calls = 0, df_offered = 0, df_accepted = 0, df_shared = 0, grouped_steps = 0; double df_ms = 0;
    // P4 B16 (docs/mimo26/P4_B16_ADAPTIVE_DRAFT.md): IE_MIMO26_DFLASH_MAX_LANES -- with this many lanes decoding or more, no lane
    // drafts or feeds the drafter's context (mimo26_lanes_draft; load default kMimo26DraftMaxLanesDefault, 0 = no cap = B5).
    // /health: the decode-step context feeds it skipped.
    uint32_t draft_max_lanes = 0;
    uint64_t df_feed_skipped = 0;
};

// P4 B5: IE_MIMO26_DRAFT_BUDGET's default (rows per group step; 0 = the B4 rule): 8, from the served A-B-A in
// docs/mimo26/P4_B5_DRAFT_BUDGET.md (2 lanes 28.4 vs 23.4 tok/s with budget 0, 4 lanes 28.5 vs 25.2; the drafted solo 22.3-22.9)
inline constexpr uint32_t kMimo26DraftBudgetDefault = 8;

// P4 B5: a lane's per-draft acceptance estimate: each verify step keeps a prefix of its drafts -- `kept` successes and one failure
// when it rejected a draft -- decayed by 3/4 per step. The prior (4 kept, 1 miss) is 0.8.
inline double mimo26_draft_acc(double kept, double miss) { return kept + miss > 0 ? kept / (kept + miss) : 0.8; }

// P4 B5: the drafts a lane offers when several lanes decode, the budget's even share: every lane of a group gets
// floor(budget / lanes_per_group) rows, its anchor and the rest drafts; capped at k_max (the drafter's 7). 0 = a plain row.
inline uint32_t mimo26_draft_k_even(uint32_t budget, uint32_t lanes_per_group, uint32_t k_max) {
    const uint32_t rows = budget / std::max<uint32_t>(1, lanes_per_group);
    return rows > 1 ? std::min(rows - 1, k_max) : 0u;
}

// P4 B5, weighted: `pool` rows for n lanes -- this lane (acc[0]) and the lanes after it in its group (acc[1..n-1]). Each keeps one
// row (its anchor); the spare rows go one at a time to the lane whose next draft is the most likely to be kept, acc^(k + 1) for
// a lane holding k drafts (acc = the per-draft acceptance, so a lane's expected kept drafts are acc + acc^2 + ...; a tie to the
// lane holding fewer drafts, then the earlier one). Returns this lane's drafts (<= k_max); 0 when the pool has no spare row for it.
inline uint32_t mimo26_draft_k_weighted(uint32_t pool, const double* acc, uint32_t n, uint32_t k_max) {
    if (n == 0 || pool <= n) return 0;
    std::vector<uint32_t> k(n, 0); std::vector<double> next(n);
    for (uint32_t i = 0; i < n; ++i) next[i] = acc[i];
    for (uint32_t spare = pool - n; spare > 0; --spare) {
        uint32_t best = n;
        for (uint32_t i = 0; i < n; ++i)
            if (k[i] < k_max && (best == n || next[i] > next[best] || (next[i] == next[best] && k[i] < k[best]))) best = i;
        if (best == n) break;
        ++k[best]; next[best] *= acc[best];
    }
    return k[0];
}

struct Mimo26Bundle {
    Mimo26Model                               model;
    Tokenizer                                 tok;
    std::vector<std::unique_ptr<sycl::queue>> queues;
    std::vector<sycl::queue*>                 qs;
    Mimo26Forward                             fwd;
    std::vector<int32_t>                      live;        // the ids at positions [0, fwd.n_pos()), in order
    std::vector<int32_t>                      eos;         // stop ids (generation_config.json eos_token_id)
    bool                                      prefix_reuse = true;
    bool                                      lookup = false;   // prompt-lookup speculation (ie serve: on; IE_MIMO26_LOOKUP=0)
    std::unique_ptr<Mimo26DFlash>             dflash;           // the checkpoint's DFlash drafter (IE_MIMO26_DFLASH=K), null = off
    uint32_t                                  dflash_k = 0;     // drafts per pass
    float                                     dflash_minp = 0.7f;   // drafts cut at the first one below this drafter probability
    // P7 (#70): what serves a prompt -- the live conversation, or another one kept in a host slot and swapped back in
    // (docs/mimo26/P7_FIX64_FIX70.md). Attached to fwd and dflash; freed in the destructor before them.
    Mimo26PrefixCache                         cache;
    // P6.2: the checkpoint's vision tower, staged in pinned host memory at load; its device block exists only while an
    // image encodes (on the first card). IE_MIMO26_VISION=0 leaves it out; a staging failure is reported per image request.
    DeviceAllocator                           vis_alloc;   // declared BEFORE vis: MimoVision frees its pinned memory through it
    MimoVision                                vis;
    bool                                      vis_ready = false;
    std::string                               vis_error = "vision tower not staged";
    uint32_t                                  image_tokens = 2048;   // per-image token budget (IE_MIMO26_IMAGE_TOKENS), <= the tower's cap
    Mimo26Serve                               serve;       // P4 B4: the lanes and the pipe's serving state (--parallel > 1)
    // Drain before the frees: an aborted generation can leave kernels in flight.
    ~Mimo26Bundle() {
        // P4 B4: the lane pipe first -- the callbacks stop resubmitting (stopping), pipe_stop drains and joins the stages (running or paused)
        { std::lock_guard<std::mutex> lk(serve.mu); serve.stopping = true; serve.pause = true; }
        if (auto e = fwd.pipe_stop(); !e.empty()) std::fprintf(stderr, "[mimo26] teardown pipe_stop: %s\n", e.c_str());
        for (auto& q : queues) if (q) { try { q->wait_and_throw(); } catch (const sycl::exception& e) { std::fprintf(stderr, "[mimo26] teardown drain: %s\n", e.what()); } }
        cache.free_all();
        if (dflash) dflash->free_all();
        fwd.free_all();
    }
};

// P6.2: the turn's text with `<|vision_start|><|image_pad|><|vision_end|>` per image -- where the server's
// kChatImageMarker sits when the markers match the images, else all of them in front (V4.1's rule).
std::string mimo26_place_images(std::string text, size_t n_images);
// The id of an image position: negative, and a function of the image's bytes and the slot -- two images never share a
// prefix in the prompt cache, the same image at the same place always does (V4.1's ds41_image_id).
int32_t mimo26_image_id(uint64_t image_hash, uint32_t slot);
uint64_t mimo26_image_hash(const std::string& bytes);    // FNV-1a over the file bytes
// The pixel budget that keeps an image at or under `tokens` image tokens (256 pixels per token after the resize).
inline uint64_t mimo26_image_max_px(uint32_t tokens) { return uint64_t(tokens) * kMimoVisFactor * kMimoVisFactor; }

// The checkpoint's chat_template.jinja in C++ (P3b): a tools block as the first system turn, ChatML turns with no
// separator between them, assistant turns as <think>{reasoning}</think>{content}{tool calls} (calls as
// <tool_call><function=NAME><parameter=K>V</parameter>...</function></tool_call>), and the generation prompt
// "<|im_start|>assistant\n" + "<think>" (thinking: the model's own first token, primed) or "<think></think>".
struct Mimo26Message { std::string role, content, reasoning, tool_calls_json; };
std::string mimo26_render_chat(const std::vector<Mimo26Message>& msgs, const std::string& tools_json, bool add_generation_prompt,
                               bool thinking, std::string& err);
// Python json.dumps(x, ensure_ascii=False) (the template's tojson): ", " and ": " separators, keys in order.
std::string mimo26_tojson(const std::string& json_text, std::string& err);

// The completion after a mimo26_render_chat prompt: reasoning up to </think> (thinking), the content, and the XML tool
// calls as an OpenAI tool_calls JSON array ("" when none); values typed by the tool's JSON schema.
// A call whose <parameter=K> has no </parameter> is REPAIRED -- the value runs to the next parameter or the function's end,
// a stray `">` dropped (MiMo at temperature 1 wrote `<parameter=action>status"></function>`); a call that cannot be
// parsed stays in `content` as text and the calls around it are still returned.
struct Mimo26Parsed { std::string reasoning, content, tool_calls_json; bool malformed_call = false; uint32_t repaired = 0; };
Mimo26Parsed mimo26_parse_completion(const std::string& text, bool thinking, const std::string& tools_json);

}  // namespace ie
