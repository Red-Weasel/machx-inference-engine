#pragma once
#include "ie/model_config.hpp"
#include <string>
#include <string_view>
namespace ie {
// Recognition by detect_arch does not imply a wired Engine runtime.
bool server_supports_arch(ModelArch arch) noexcept;
bool server_streams_experts(ModelArch arch) noexcept;
// Metadata only; never initializes an allocator or uploads model weights.
// Throws std::runtime_error when a supplied GGUF cannot be opened.
std::string server_capabilities_json(const std::string& model_path = {});
std::string server_reasoning_error(const std::string& model_path, std::string_view effort);
}
