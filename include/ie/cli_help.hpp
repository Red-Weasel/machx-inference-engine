#pragma once
// `ie --help`, `ie help <command>`, `ie <command> --help`: the help text. Printing it loads nothing.
// The flags are parse_launch_options' (ie/serve_options.hpp); tests/unit/cli_help_test.cpp fails when the parser
// accepts a flag that the run/serve help does not name.
#include "ie/serve_options.hpp"
#include <string>
#include <vector>

namespace ie {

inline const char* cli_help_overview() {
    return
"ie -- the MachX inference engine\n"
"\n"
"usage: ie <command> [arguments]\n"
"\n"
"Commands:\n"
"  serve <model> [flags]         OpenAI-compatible HTTP server (/v1/chat/completions)\n"
"  run   <model> [flags]         interactive chat in the terminal\n"
"  supervise --config <layout>   several servers (one per layout entry, each on its own cards)\n"
"                                behind one endpoint that routes by the request's \"model\"\n"
"  capabilities [model]          what the server supports, as JSON (no GPU load)\n"
"  preflight <model.gguf> [...]  will this model load? (metadata only, no GPU upload)\n"
"  cards                         list the GPUs --cards can choose (no GPU load)\n"
"  bench <model.gguf>            run ie-bench from the sibling tools/ build\n"
"  pull  <name|hf-repo> [file]   download a GGUF (ie pull --list for curated names)\n"
"  import <hf_dir> <out.gguf> <tokenizer_ref.gguf>\n"
"                                convert an AWQ/GPTQ/EXL3 checkpoint to GGUF\n"
"  help [command]                this text, or one command's help\n"
"\n"
"<model> is a .gguf file (for a split GGUF, the ...-00001-of-NNNNN.gguf shard), or a\n"
"DeepSeek-V4.1-Flash / MiMo-V2.6 safetensors checkpoint directory.\n"
"\n"
"`ie <command> --help` (or -h) prints that command's flags, defaults and the environment\n"
"variables that matter. `ie serve --config <layout.json>` reads the flags from a layout\n"
"file (docs/serve_config.md).\n";
}

// The launch flags (run, serve; bench and preflight accept the same set). Grouped; defaults are today's.
inline const char* cli_help_launch_flags() {
    return
"Model and memory:\n"
"  --ctx N                 context window in tokens, 9..2147483647 (default 8192)\n"
"  --int8-kv               int8 KV cache (not with --parallel > 1; glm5next refuses it)\n"
"  --prefill-chunk N       tokens per prefill forward, >= 1 (default 256). On DeepSeek-V4 it is the\n"
"                          largest T any forward submits.\n"
"  --vram-reserve-gib F    GiB of VRAM the automatic expert tier leaves free per card, 0..24\n"
"                          (mimo_v2 and deepseek41 only; default 1.5 for mimo_v2, 6 for deepseek41;\n"
"                          raise it for very long contexts). Sets IE_MIMO26_VRAM_RESERVE_GIB and\n"
"                          IE_DS41_VRAM_RESERVE_GIB.\n"
"  --threads N             CPU threads, 1..1024 (default: the model's own CPU team size).\n"
"                          Sets OMP_NUM_THREADS and IE_CPU_THREADS.\n"
"  --no-prompt-cache       do not reuse a cached prompt prefix across turns (default: reuse)\n"
"\n"
"GPUs:\n"
"  --cards LIST            which physical cards this process may use, e.g. 1 or 0,1 (default: all).\n"
"                          Card N = the N-th discrete GPU in PCI address order (the integrated GPU is\n"
"                          never a card); `ie cards` lists them. The other cards are hidden from the\n"
"                          process (ONEAPI_DEVICE_SELECTOR); cannot be combined with a restricting\n"
"                          ONEAPI_DEVICE_SELECTOR or with ZE_AFFINITY_MASK.\n"
"  --gpus N                how many of the visible cards to split one model across, 0..64\n"
"                          (default 0 = automatic: the planner picks one card or a split from free\n"
"                          VRAM). N > 1: tensor-parallel for dense archs, layer split for\n"
"                          Qwen3-Next-80B; DeepSeek-V4 defaults to every card (expert tensor\n"
"                          parallel); glm5next takes 1 or 2. Must not exceed the --cards count.\n"
"\n"
"Serving (serve):\n"
"  --host H                bind address (default 127.0.0.1)\n"
"  --port P                1..65535 (default 11435)\n"
"  --parallel N            concurrent generations, 1..16 (default 1). >1: N request lanes that decode\n"
"                          together through the card pipe on mimo_v2, deepseek41, qwen4exp (Flash-Next),\n"
"                          qwen35moe (the 35B-A3B crown split) and the qwen35 27B split; the last four\n"
"                          need two cards. IE_Q4E_LANES=0 / IE_Q35MOE_LANES=0 / IE_QWEN35_LANES=0 turn\n"
"                          them off (the 27B then batches decode over N slot banks, as it also does\n"
"                          with --spec); qwen4exp on one card or with\n"
"                          IE_Q4E_LANES=0 time-slices; every other arch, glm5next and deepseek4\n"
"                          included, runs whole generations in FIFO turns. Images need --parallel 1\n"
"                          on mimo_v2, qwen4exp and deepseek4 (deepseek41 takes them on its lanes).\n"
"                          Not with --int8-kv. The load refuses, with the numbers, a count whose lanes do\n"
"                          not fit (--slot-ctx sizes them); the 27B split path batches at most 16.\n"
"  --slot-ctx N            0 or 9..--ctx (default 0). The lanes: the positions of lanes 1..N-1 (lane 0\n"
"                          has --ctx; 0 = min(--ctx, 32768)); the load refuses when they do not fit\n"
"                          (mimo_v2, deepseek41 and qwen4exp take that VRAM from the expert tier). The\n"
"                          27B split: lanes 1..N-1 (or each decode bank) hold 0 = min(--ctx, 65536); a\n"
"                          bank request whose prompt + max_tokens exceeds it is time-sliced instead.\n"
"                          Other archs ignore it.\n"
"  --max-queue N           requests allowed to WAIT for a generation slot, 0..1024 (default 8);\n"
"                          beyond that the server answers HTTP 429 at once\n"
"\n"
"Speculative decoding (Qwen3.6-27B kQwen35Dense unless noted; always lossless greedy):\n"
"  --spec                  MTP self-speculative GREEDY decode (~1.1-1.2x on Q8_0 MTP heads; it\n"
"                          REGRESSES on Q4_K_M). `ie run` forces temperature 0 with it.\n"
"  --spec-k K              draft length, 1..64 (default 2)\n"
"  --spec-head FILE        gemma4 MTP draft head GGUF (default: the first mtp-*.gguf beside the model)\n"
"  --spec-draft FILE       separate dspark drafter GGUF (target-conditioned)\n"
"\n"
"Sampling defaults (each request may override them):\n"
"  --temp F                0..2 (default 0.7)\n"
"  --top-k N               0..1024 (default 40; 0 = the sampler ceiling of 1024)\n"
"  --top-p F               (0, 1] (default 0.95)\n"
"  --min-p F               0..1 (default 0)\n"
"  --repeat-penalty F      (0, 10] (default 1 = off)\n"
"  --repeat-last-n N       0..512 (default 64); the history penalties share this window over\n"
"                          prompt + output; 0 disables all history penalties\n"
"  --presence-penalty F    -2..2 (default 0)\n"
"  --frequency-penalty F   -2..2 (default 0)\n"
"  --seed N                0..2^64-1 (default 0 = random)\n"
"  --max-tokens N          completion cap (default 16384; 0 = until the context is full)\n"
"  --stop STRING           stop string, repeat up to 4 times\n"
"  --thinking on|off       reasoning on or off by default (default on, or off with IE_SERVE_NO_THINK)\n"
"  --reasoning-effort L    default effort; the levels depend on the model (see ie capabilities <model>)\n"
"\n"
"Layout file:\n"
"  --config FILE           read the flags from a JSON layout (docs/serve_config.md); flags given on\n"
"                          the command line override the file, the <model> argument overrides its\n"
"                          \"model\". A layout with several servers runs under `ie supervise`.\n"
"  --server NAME           with --config: run the layout's server of that name (how `ie supervise`\n"
"                          starts each one); /v1/models then reports NAME, with \"root\" = the model id\n";
}

inline const char* cli_help_env() {
    return
"Environment (the variables that matter for serving; the engine reads ~450 IE_* tuning\n"
"variables in all, documented beside their code under docs/):\n"
"  Devices:   ONEAPI_DEVICE_SELECTOR / ZE_AFFINITY_MASK hide devices from the runtime (--cards sets\n"
"             the first); IE_GPU_FILTER=<name> re-points the GPU name filter (default \"B70\";\n"
"             the deepseek41 and mimo_v2 runtimes take every Level Zero GPU named \"Arc\");\n"
"             DS4_GPU / DS4_TP_GPUS pick DeepSeek-V4 ordinals\n"
"  Defaults:  IE_SERVE_TEMP, IE_SERVE_TOP_K, IE_SERVE_TOP_P, IE_SERVE_MIN_P, IE_SERVE_REPEAT_PENALTY,\n"
"             IE_SERVE_REPEAT_LAST_N, IE_SERVE_PRESENCE_PENALTY, IE_SERVE_FREQUENCY_PENALTY,\n"
"             IE_SERVE_SEED, IE_SERVE_MAX_TOKENS, IE_SERVE_REASONING_EFFORT, IE_SERVE_NO_THINK\n"
"             (set) -- server sampling defaults; the flags above override them\n"
"  Cache:     IE_NO_PROMPT_CACHE (set) = off; IE_PROMPT_CACHE_MAX_ENTRIES;\n"
"             IE_MIMO26_PROMPT_CACHE=0 / IE_MIMO26_PROMPT_CACHE_GIB;\n"
"             IE_DS41_PROMPT_CACHE=0 / IE_DS41_PROMPT_CACHE_GIB / IE_DS41_PROMPT_CACHE_DISK_GIB /\n"
"             IE_DS41_PROMPT_CACHE_DIR (0 = no disk entries)\n"
"  Speculation: IE_MIMO26_DFLASH=K drafter length (default 7 when the checkpoint has dflash/;\n"
"             0 = off); IE_MIMO26_LOOKUP=1|0 prompt lookup (default on only without DFlash);\n"
"             IE_DS41_LOOKUP=0 turns deepseek41 prompt lookup off (default on)\n"
"  Vision:    IE_MIMO26_VISION=0, IE_DS41_VISION=0 leave the vision tower out; IE_MMPROJ=<file>\n"
"             names the qwen4exp projector\n"
"  Experts:   IE_MIMO26_STATIC / IE_MIMO26_PINNED / IE_MIMO26_STREAM pin tier sizes;\n"
"             IE_MIMO26_RANKING / IE_DS41_RANKING name a residency ranking\n"
"  Server:    IE_EXIT_ON_DEVICE_LOST=0 stays up after a lost device (default: exit 75);\n"
"             IE_IDLE_SPIN_WATCHDOG=0 / IE_IDLE_SPIN_CORES; GLIBC_TUNABLES (when unset, serve\n"
"             pins the malloc mmap threshold at 256 KiB)\n"
"  run:       IE_PERF (set) prints prefill/decode rates after each reply\n";
}

// "" for a command without help (unknown).
inline std::string cli_help(const std::string& cmd) {
    if (cmd == "serve")
        return std::string(
"usage: ie serve <model> [flags]\n"
"       ie serve [<model>] --config <layout.json> [flags]\n"
"\n"
"Load <model> and serve the OpenAI chat API until /admin/shutdown (loopback only), SIGTERM or\n"
"SIGINT. Endpoints: POST /v1/chat/completions, GET /v1/models, GET /health (503 once a device\n"
"is lost or while stopping), GET /props (n_ctx, slots, residency, vision), POST /admin/shutdown,\n"
"POST /v1/embeddings (501).\n"
"\n") + cli_help_launch_flags() + "\n" + cli_help_env();
    if (cmd == "run")
        return std::string(
"usage: ie run <model> [flags]\n"
"       ie run [<model>] --config <layout.json> [flags]\n"
"\n"
"Load <model> and chat in the terminal: /reset clears the history, /quit exits. The serving\n"
"flags (--host, --port, --parallel, --slot-ctx, --max-queue) are accepted and validated but\n"
"a terminal chat has no use for them.\n"
"\n") + cli_help_launch_flags() + "\n" + cli_help_env();
    if (cmd == "supervise")
        return
"usage: ie supervise --config <layout.json> [--host H] [--port P]\n"
"\n"
"Start one `ie serve` child per server of the layout (each as `ie serve --config <layout>\n"
"--server <name>`, pinned to its \"cards\", on its own \"port\") and serve ONE OpenAI endpoint\n"
"in front of them (docs/serve_config.md, Supervisor):\n"
"  POST /v1/chat/completions, /v1/completions  routed by the body's \"model\": a server name, or\n"
"                          its model id when only one server has it; no \"model\" = the layout's\n"
"                          \"default\" server (400 without one); unknown = 404 naming the models.\n"
"                          The reply (SSE included) is passed through byte for byte.\n"
"  GET /v1/models          every server by name (\"root\" = model id, \"status\")\n"
"  GET /health             per-server state and health; 200 when at least one server is ready\n"
"  GET /props?model=NAME   that server's /props (no model: the default server's)\n"
"  POST /admin/shutdown    (loopback only) stop everything, like SIGTERM / SIGINT\n"
"\n"
"  --config FILE           the layout: every server needs \"name\", \"model\", \"cards\", \"port\";\n"
"                          top-level \"default\", \"host\", \"port\" configure the front endpoint;\n"
"                          per-server \"restart\": \"none\" (default) or \"on-failure\" (up to 3 times,\n"
"                          never after exit 75 = device lost)\n"
"  --host H                front bind address (default: the layout's \"host\", else 127.0.0.1)\n"
"  --port P                front port (default: the layout's \"port\", else 11435)\n"
"\n"
"Shutdown sends each child its own /admin/shutdown (a loading child first finishes loading) and\n"
"waits for it to exit; a child is NEVER signalled or killed -- one that does not stop is\n"
"reported and left running. A second SIGINT/SIGTERM exits the supervisor at once, children left\n"
"running. Card locks: the supervisor holds <dir>/ie-card-N.lock for every child's cards\n"
"(IE_CARD_LOCK_DIR, default /tmp) and refuses to start when another engine process holds one.\n"
"Exit 0 = every child stopped cleanly, 1 = a child failed or was left running, 2 = not started.\n";
    if (cmd == "capabilities")
        return
"usage: ie capabilities [model]\n"
"\n"
"Print what the server supports as JSON -- architecture, launch and sampling defaults, features\n"
"(prompt cache, speculation, vision, int8 KV), load options, memory planner, reasoning -- for\n"
"<model> (a .gguf or a checkpoint directory; its metadata only) or, without one, generically.\n"
"Loads nothing onto a GPU. Exit 0, or 2 when the model cannot be read.\n";
    if (cmd == "preflight")
        return
"usage: ie preflight <model.gguf> [--ctx N] [--gpus N] [--int8-kv] [--cards LIST]\n"
"\n"
"Metadata-only load check: opens the GGUF, reports tensor dtypes, the architecture and the\n"
"VRAM plan for the visible cards, uploads nothing. Exit 0 = will load, 3 = will not,\n"
"1 = cannot open the file. --ctx (default 8192), --gpus (default 0 = automatic), --int8-kv and\n"
"--cards shape the plan; every other run/serve flag is accepted and validated but unused.\n"
"deepseek4: the VRAM verdict does not apply (experts stream from host RAM); exit is 0 and\n"
"the real check happens at load.\n";
    if (cmd == "cards")
        return
"usage: ie cards\n"
"\n"
"List the Level Zero GPUs: card number (for --cards and the layout's \"cards\"), PCI address,\n"
"name. The integrated GPU is listed without a card number. Loads nothing; ignores\n"
"ONEAPI_DEVICE_SELECTOR (it lists what --cards would choose from).\n";
    if (cmd == "bench")
        return
"usage: ie bench <model.gguf> [flags]\n"
"\n"
"Run ie-bench (build/tools/ie-bench, else ie-bench on PATH) as `ie-bench --gguf <model.gguf>`.\n"
"The run/serve flags are validated; only their environment effects reach ie-bench\n"
"(--threads, --vram-reserve-gib, --cards). For ie-bench's own options run it directly.\n";
    if (cmd == "pull")
        return
"usage: ie pull <name>                   a curated, engine-validated model\n"
"       ie pull <hf-repo> <file.gguf>     any GGUF on Hugging Face\n"
"       ie pull <hf-repo>                 list the .gguf files in a repo\n"
"       ie pull --list                    the curated names\n"
"\n"
"Delegates to the ie-pull helper (next to ie, scripts/ie-pull, else PATH). Downloads into\n"
"$IE_MODELS (default ~/models) and prints the path to pass to `ie serve`. Uses the hf CLI\n"
"when present, else curl for public repos.\n";
    if (cmd == "import")
        return
"usage: ie import <hf_dir> <out.gguf> <tokenizer_ref.gguf>\n"
"\n"
"Convert an AWQ / GPTQ / EXL3 Hugging Face checkpoint directory to GGUF. The reference GGUF\n"
"supplies the tokenizer KVs of the model family. No GPU. Exit 0 on success, 1 on failure.\n";
    if (cmd == "help")
        return
"usage: ie help [command]\n"
"\n"
"Without a command: the command list. With one: that command's help (same as\n"
"`ie <command> --help`).\n";
    return {};
}

// True when the arguments after the command ask for help: a --help / -h that is not the value of a flag
// (`--stop --help` is a stop string).
inline bool cli_wants_help(const std::vector<std::string>& args) {
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--help" || args[i] == "-h") return true;
        if (args[i].rfind("--", 0) == 0 && !launch_flag_is_boolean(args[i])) ++i;   // skip the flag's value
    }
    return false;
}

}  // namespace ie
