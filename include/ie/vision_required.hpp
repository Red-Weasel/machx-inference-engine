// include/ie/vision_required.hpp -- a model with an image path and its image encoder at the load (2026-11-11).
//
// When the encoder file is there, the encoder is part of the load: uploaded before the memory budgets are fixed, never at the
// first image. When it is NOT there, two behaviours:
//   default               the load goes on text-only and says so on stderr; image requests are refused by name (/props "vision"
//                         reports it). Some people run without images.
//   IE_VISION_REQUIRED=1  the load FAILS and names the files to add. For a deployment where a model that cannot see is a
//                         fault, not a mode. v0.2.40 made this the only behaviour; that was wrong for a default.
// Used by the Flash-Next (qwen4exp) load. The other families keep their own rules (ie/q4e_lanes.hpp's *_vision_refusal).
#pragma once

#include <cstdlib>
#include <string>
#include <string_view>

namespace ie {

inline bool vision_required_env() {
    const char* v = std::getenv("IE_VISION_REQUIRED");
    return v && *v == '1';
}

// The load-time verdict. `problem` = why the encoder is not there ("" = it is loaded). Returns the load's error ("" = the load
// goes on); when it goes on WITHOUT the encoder, `warning` holds the line to print.
inline std::string vision_required_error(std::string_view family, std::string_view problem, std::string_view fix, bool required,
                                         std::string* warning = nullptr) {
    if (warning) warning->clear();
    if (problem.empty()) return {};
    if (required)
        return std::string(family) + ": IE_VISION_REQUIRED=1 and " + std::string(problem) + " -- " + std::string(fix);
    if (warning)
        *warning = "[vision] " + std::string(family) + ": " + std::string(problem) + " -- loading TEXT-ONLY, image requests will be "
                   "refused. To read images: " + std::string(fix) + ". (IE_VISION_REQUIRED=1 makes this load fail instead.)";
    return {};
}

}  // namespace ie
