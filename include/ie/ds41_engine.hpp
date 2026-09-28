// include/ie/ds41_engine.hpp — the DeepSeek-V4.1-Flash bundle behind the Engine (Phase 12,
// docs/deepseek41/31): the model, its engram tables, the tokenizer.json tokenizer, one in-order
// queue per Arc card, and the resident two-card runtime. Complete here so Engine's destructor
// (engine.cpp) can destroy the unique_ptr; built and used in src/engine/ds41_engine.cpp.
// One request at a time at --parallel 1 (the path the engine always had); --parallel N > 1 serves N requests at once on N
// lanes (P4 B6b, Ds41Serve below, docs/deepseek41/P4_B6B_SERVE.md).
#pragma once
#include "ie/deepseek41.hpp"
#include "ie/deepseek41_engram.hpp"
#include <cstdio>

#include "ie/allocator.hpp"
#include "ie/deepseek41_dspark.hpp"
#include "ie/deepseek41_forward.hpp"
#include "ie/deepseek41_generate.hpp"
#include "ie/ds41_vision.hpp"
#include "ie/ngram_draft.hpp"
#include "ie/tokenizer.hpp"

#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace ie {

// P4 B6b (docs/deepseek41/P4_B6B_SERVE.md; MiMo B4's design, docs/mimo26/P4_B4_SERVE.md): --parallel N > 1. Every request owns
// one LANE of Ds41Forward (its caches, positions, look-back and live prefix checkpoints, B6a) from admission until it has
// read its outcome; an idle lane keeps its conversation for a later prompt's prefix step. Steps run through the lane pipe
// (Ds41Forward::pipe_*, B6a: card 0 runs one lane's step while card 1 runs another's): the pipe's done callback -- on the last
// card's stage thread -- samples a lane's rows on the host (a prompt-lookup verify step's rows judged there, the lane rolled
// back with pipe_rollback), commits the ids to the lane's outbox and submits the lane's next step; the request's own thread
// only takes ids from the outbox, decodes them to text and runs the server's callback. The prefix step, the prompt's end
// (the checkpoints, the think tag, the disk entry) and a prompt with images need the serial API, so a request takes the
// SERIAL TURN for them: with `pause` set the callbacks park their lanes instead of resubmitting, pipe_pause() drains the
// steps in flight (the stage threads stay), the holder works alone on the cards, and releasing the turn resumes the pipe with
// every parked lane's step -- unless another request waits for the turn, which then takes it with the pipe still paused.
// A prompt that arrives while every other lane is idle runs its prefill in the turn with the cards pipelined over its chunks
// (forward_pipelined, as at --parallel 1), handing the rest to the lane pipe at a chunk end as soon as another request
// waits. One turn at a time; `mu` guards everything below. The turn's device work runs on one persistent serial worker
// thread (below), with `mu` released.
struct Ds41Serve {
    struct Lane {
        enum class Phase : uint8_t { kIdle, kPrefill, kPromptReady, kDecode, kDone };
        bool     busy = false;                    // owned by a request
        uint32_t cap = 0;                         // position capacity: lane 0 --ctx, the others --slot-ctx
        Phase    phase = Phase::kIdle;
        const std::vector<int32_t>* prompt = nullptr;   // the request's ids (its thread owns them; valid while busy)
        Ds41PrefillPlan plan; uint32_t chunk_at = 0;    // the prefill plan and its next chunk; then one-row steps [plan.tail, plan.Tp)
        uint32_t reused = 0;                      // prompt positions served from the prefix cache
        uint32_t pos = 0;                         // positions in the lane's caches (the next step's pos0)
        uint32_t max_new = 0, n_new = 0;
        int32_t  next = -1;                       // the sampled id whose forward is due
        std::vector<int32_t> rows;                // the step in flight: its prefill rows' count in `pend`, or [next, drafts...]
        uint32_t pend = 0;
        Ds41SampleParams sp; uint64_t rng = 0; std::vector<int32_t> recent;   // the sampler's state and repetition window
        bool     lookup = false; Ds41NgramIndex idx;   // prompt-lookup speculation for this lane (Phase 58)
        std::vector<int32_t> out;                 // every committed id (the repetition rule)
        std::vector<int32_t> outbox;              // committed ids the request thread has not taken
        std::vector<float> first_logits, scratch; // the prompt's last row (kPromptReady); the sampler's copy of a row
        std::string finish;                       // "" while running; "stop" / "length" / "repetition" / "abort" / "error: ..."
        bool     want_stop = false;               // the request thread's cancel: honoured at the lane's next completion
        bool     lost = false;                    // the lane's state must be forgotten at the release (a pipe error)
        bool     parked = false;                  // a step is due and not submitted (the pipe was paused, or the lane is new)
        uint64_t tick = 0;                        // the lane's last release (larger = more recent): the LRU idle lane is taken first
        uint32_t last_tp = 0;                     // the planned end of the last prompt that reached its end here (0: none, or cut since)
        std::chrono::steady_clock::time_point t_sub{}, t_first{};
        double   early_ms = 0;                    // the first 100 committed tokens' time (the server's pace line)
        uint32_t steps = 0, chunks = 0, passes = 0, pass_rows = 0, accepted = 0, plain = 0;
        double   cb_ms = 0;                       // (diagnostic) the done callbacks' time for this request
    };
    std::vector<Lane>       lanes;                // empty at --parallel 1 (the serial path)
    std::mutex              mu;
    std::condition_variable cv;
    bool     piping = false, pause = false, turn_busy = false, stopping = false;   // piping: the pipe takes steps (started, not paused)
    uint32_t turn_waiters = 0;
    uint64_t tick = 0;
    bool     lookup = false;                      // prompt-lookup speculation for every lane (Ds41Bundle::lookup)
    Ds41LookupPolicy lp;
    // IE_DS41_MIX_CHUNK: a prefill chunk's rows while another lane is busy. Default = the forward's chunk (no cap): MiMo B4
    // measured a MiMo chunk's cost as mostly per chunk (a 512-row cap doubled an arrival's TTFT); a V4.1 chunk also streams
    // most of its layers' experts whatever its rows (assumed from docs/deepseek41/83, not measured for the mix)
    uint32_t mix_chunk = 0;
    // /health: the decode steps (submit to callback) as an exponential average, and their rows; ids committed (every lane)
    double   step_ms = 0, rows_per_step = 0;
    uint64_t steps = 0, tokens = 0;
    uint64_t turns = 0; double paused_ms = 0;     // serial turns taken, and the time the pipe stayed paused for them (the turn's work)
    uint64_t handovers = 0;                       // turns released straight to a waiting request (the pipe not resumed in between)
    std::chrono::steady_clock::time_point turn_t0{};
    // The serial turn's device work runs on ONE persistent thread, not on the HTTP request threads: a thread that runs a forward
    // runs the engram gather's OpenMP region and keeps that team for good (the adca570 lesson), so request threads taking turns
    // would each grow the process by a team. The turn holder posts `wjob` and waits for `wdone`; `wmu` guards these.
    std::thread             worker;
    std::mutex              wmu;
    std::condition_variable wcv;
    const std::function<void()>* wjob = nullptr;
    bool                    wdone = false, wstop = false;
    std::string             werr;                 // the job threw: what (the request fails with it)
};

struct Ds41Bundle {
    DeepSeek41Model                             model;
    Ds41EngramTables                            tables;
    Tokenizer                                   tok;
    std::vector<std::unique_ptr<sycl::queue>>   queues;
    std::vector<sycl::queue*>                   qs;
    Ds41Forward                                 fwd;
    Ds41Drafter                                 drafter;   // DSpark P4 (docs/deepseek41/58): built under IE_DS41_SPEC=1
    // Phase 48 (docs/deepseek41/88): IE_DS41_PROFILE_OUT -- the DECODE routing of every request served, summed per
    // (layer, expert) and rewritten as a residency ranking after each request, so a client's own traffic can be profiled
    // Phase 57 (docs/deepseek41/96): the checkpoint's vision tower, transient on card 0 (its device block lives only
    // while an image is encoded). `vis_error` says why image requests are refused when it is not ready.
    DeviceAllocator                             vis_alloc;
    Ds4Vision                                   vis;
    bool                                        vis_ready = false;
    std::string                                 vis_error = "the vision tower is not loaded";
    // Phase 58-59 (docs/deepseek41/97-98): prompt-lookup speculation for served requests, ON unless IE_DS41_LOOKUP=0
    bool                                        lookup = true;
    std::string                                 profile_out;
    std::vector<std::vector<uint64_t>>          profile_acc;
    uint64_t                                    profile_steps = 0;
    Ds41Serve                                   serve;     // P4 B6b: the lanes and the pipe's serving state (--parallel > 1)
    // Drain BEFORE any free: the drafter's and the runtime's device memory can still be referenced by
    // kernels an aborted generation left in flight (see Ds41Forward::free_resident). The final wait is
    // kept as a belt-and-braces flush of anything the frees themselves submitted.
    ~Ds41Bundle() {
        // P4 B6b: the lanes' serial worker (idle: no request is left), then the lane pipe -- the callbacks stop resubmitting
        // (stopping), pipe_stop drains and joins the stages (running or paused)
        { std::lock_guard<std::mutex> lk(serve.mu); serve.stopping = true; serve.pause = true; }
        { std::lock_guard<std::mutex> lk(serve.wmu); serve.wstop = true; }
        serve.wcv.notify_all();
        if (serve.worker.joinable()) serve.worker.join();
        if (auto e = fwd.pipe_stop(); !e.empty()) std::fprintf(stderr, "[deepseek41] teardown pipe_stop: %s\n", e.c_str());
        for (auto& q : queues) if (q) { try { q->wait_and_throw(); } catch (const sycl::exception& e) { std::fprintf(stderr, "[deepseek41] teardown drain: %s\n", e.what()); } }
        if (drafter.ready() && !qs.empty()) drafter.free(*qs.back());
        fwd.free_resident();
        for (auto& q : queues) if (q) { try { q->wait_and_throw(); } catch (const sycl::exception&) {} }
    }
};

}  // namespace ie
