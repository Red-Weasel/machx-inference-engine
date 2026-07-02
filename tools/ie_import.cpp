// tools/ie_import.cpp — convert a HF AWQ/GPTQ/EXL3 checkpoint to a native GGUF.
//   ie-import <hf_dir> <out.gguf> <tokenizer_ref.gguf>
// The ref GGUF supplies the tokenizer KVs (any same-vocab GGUF of the family).
// The quant_method in config.json selects the importer (awq/gptq vs exl3).
#include "ie/hf_import.hpp"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <hf_dir> <out.gguf> <tokenizer_ref.gguf>\n", argv[0]);
        return 2;
    }
    const std::string hf_dir = argv[1], out_gguf = argv[2], tok_ref = argv[3];

    // Peek at config.json to route to the right importer.
    std::string method = "awq", arch;
    {
        std::ifstream cf(hf_dir + "/config.json");
        if (cf) {
            const std::string cfg((std::istreambuf_iterator<char>(cf)),
                                  std::istreambuf_iterator<char>());
            ie::HfModelMeta meta;
            if (ie::parse_hf_config(cfg, meta).empty() && meta.quantized) {
                method = meta.quant.quant_method;
                arch   = meta.arch;
            }
        }
    }

    std::string log, err;
    if (method == "exl3")
        err = (arch == "qwen3_next" || arch == "qwen3next")
            ? ie::import_exl3_qwen3next_to_gguf(hf_dir, out_gguf, tok_ref, &log)
            : ie::import_exl3_to_gguf(hf_dir, out_gguf, tok_ref, &log);
    else
        err = ie::import_awq_to_gguf(hf_dir, out_gguf, tok_ref, &log);
    std::fputs(log.c_str(), stdout);
    if (!err.empty()) {
        std::fprintf(stderr, "import failed: %s\n", err.c_str());
        return 1;
    }
    std::printf("OK: %s\n", out_gguf.c_str());
    return 0;
}
