// include/ie/exit_codes.hpp -- the process exit codes `ie serve` uses to tell a supervisor what happened, in one small
// header: the supervisor (include/ie/supervisor.hpp) reads them to decide on a relaunch, and the model code that ends the
// process on a wedged card (src/model/deepseek41_forward.cpp's link probe) raises them. Kept apart from supervisor.hpp so
// the V4.1 numerics-key source list (src/model/deepseek41_numerics_inputs.txt, closed under #include) carries only this.
#pragma once

namespace ie::sup {

inline constexpr int kDeviceLostExit = 75;   // ie serve's exit code after a lost device (IE_EXIT_ON_DEVICE_LOST)
inline constexpr int kCardWedgedExit = 76;   // ie serve's exit code when a card's first copy never completed (a wedged driver
                                             // queue: deepseek41_forward.cpp's link probe, 2026-10-27); a relaunch is futile
                                             // until a reboot

}  // namespace ie::sup
