#pragma once
// `ie supervise --config <layout.json>` (docs/serve_config.md, "Supervisor"): one `ie serve` child per layout server,
// each pinned to its cards on its own port, behind ONE OpenAI-compatible front endpoint that routes by the request's
// "model" field.
//
//   - Children are started as `ie serve --config <layout> --server <name>` in their own process group (a terminal ^C
//     reaches only the supervisor); its stdout/stderr lines reach the supervisor's stderr prefixed "[<name>] ".
//     Readiness = the child's /health answers 200 (its listener starts after the load).
//   - A child that exits is reported (code / signal) and the others keep serving; "restart": "on-failure" restarts it
//     (at most kMaxRestarts times; never after exit 75 = device lost, and never while stopping). Default: no restart.
//   - Shutdown (SIGTERM, SIGINT, POST /admin/shutdown from loopback): new requests get 503; each child gets its own
//     POST /admin/shutdown -- a child still loading first gets time to finish loading -- and the supervisor waits for
//     it to exit. It never signals a child: a child that has not stopped within the timeout is reported and left
//     running (killing a process that holds a model can wedge the GPU). A second signal exits at once, children left.
//   - Front: /v1/models (every server by name, "root" = the model id), /v1/chat/completions and /v1/completions
//     proxied to the routed child with the response streamed through unchanged, /health (aggregate), /props?model=,
//     POST /admin/shutdown (loopback only).
//
// Host-only (no SYCL, no model): the pieces below are unit-tested with fake children (tests/unit/supervisor_test.cpp).
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ie::sup {

struct ServerSpec {
    std::string name;                                      // the layout's name (what clients put in "model")
    std::string model_id;                                  // the underlying model id (file name without .gguf)
    std::string connect_host = "127.0.0.1";                // where the supervisor reaches the child (loopback)
    int port = 0;
    std::vector<uint32_t> cards;
    std::vector<std::string> argv;                         // argv[0] = the executable
    std::vector<std::pair<std::string, std::string>> env;  // set in the child's environment (over the inherited one)
    bool restart_on_failure = false;
    unsigned slots = 12;                                   // parallel + max_queue: sizes the front's worker pool
};

struct Options {
    std::string host = "127.0.0.1";
    int port = 11435;
    int default_index = -1;             // the server a request without "model" goes to (-1 = none: 400)
    double poll_s = 1.0;                // child /health polling period
    double ready_wait_s = 1800;         // shutdown: how long a loading child may take to become ready
    double stop_wait_s = 600;           // shutdown: how long a child may take to exit after /admin/shutdown
    bool take_card_locks = true;        // per-card locks (ie/card_lock.hpp)
};

inline constexpr int kMaxRestarts = 3;
inline constexpr int kDeviceLostExit = 75;   // ie serve's exit code after a lost device (IE_EXIT_ON_DEVICE_LOST)

// ---- routing (pure) --------------------------------------------------------------------------------------------------
struct Route {
    int index = -1;       // the server, when status == 200
    int status = 200;     // 400 (bad body / no model and no default / ambiguous id), 404 (unknown model)
    std::string error;    // OpenAI-shaped error JSON when status != 200
};
// `model` = nullptr: the request has no "model" field. Match by name first, then by model id when exactly one
// server has it.
Route route_model(const std::vector<ServerSpec>& servers, int default_index, const std::string* model);
// The request body's "model" field routed (the body itself is forwarded unchanged).
Route route_request(const std::vector<ServerSpec>& servers, int default_index, const std::string& body);

// ---- state and the aggregate views (pure) ------------------------------------------------------------------------------
enum class ChildState { Loading, Ready, Unhealthy, Stopping, Exited, Stopped, LeftRunning };
const char* state_name(ChildState s);

struct ChildView {
    ChildState state = ChildState::Loading;
    int pid = -1;
    int exit_code = -1;          // Exited/Stopped: the exit code, or -1
    int exit_signal = 0;         // Exited: the terminating signal, or 0
    int restarts = 0;
    std::string health;          // the child's last /health body (JSON), "" when none
    double since_s = 0;          // seconds since the child was started
};

// /health: {"status": ok|degraded|loading|unavailable|stopping, "inflight", "queued", "servers": {name: {...}}}.
// 200 when every server is ready ("ok") or at least one is ("degraded"); 503 otherwise and while stopping.
std::pair<int, std::string> aggregate_health(const std::vector<ServerSpec>& servers, const std::vector<ChildView>& views,
                                             bool stopping, int default_index);
// /v1/models: every server by name, with "root" = its model id and its state.
std::string models_list_json(const std::vector<ServerSpec>& servers, const std::vector<ChildView>& views);

// ---- the supervisor ----------------------------------------------------------------------------------------------------
// Blocks until shut down. Returns 0 when every child's process exited with code 0; 1 when a child is still running or
// a child's last exit was a signal or a non-zero code (in its requested shutdown too); 2 when the supervisor could not
// start (lock held, bind failure; nothing was started).
int run_supervisor(const std::vector<ServerSpec>& servers, const Options& opt);

}  // namespace ie::sup
