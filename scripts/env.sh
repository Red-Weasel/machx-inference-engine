# Source this file to set up a build shell.
#   source scripts/env.sh
#
# Adds locally-installed cmake/ninja to PATH and initializes oneAPI with the
# compiler pinned to IE_ONEAPI_COMPILER (default 2026.1).

# Find the project root regardless of cwd.
_IE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

# Local tooling (cmake, ninja) installed under ~/.local/bin
case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) export PATH="$HOME/.local/bin:$PATH" ;;
esac

# oneAPI — DPC++ icpx + Level Zero + SYCL runtime, the compiler PINNED to the
# one the build dirs are configured with (CMakeCache: compiler/2026.1/bin/icpx).
# Plain setvars.sh takes the newest install: oneAPI 2026.2 (installed
# 2026-11-07 17:15) then links icpx 2026.1's libsycl against 2026.2's
# libur_loader (undefined urKernelSetArg*@LIBUR_LOADER_0.12 -- no build links)
# and runs the installed ie on a runtime no gate used. Re-pins even when the
# caller's shell already sourced a different oneAPI.
IE_ONEAPI_COMPILER="${IE_ONEAPI_COMPILER:-2026.1}"
if [ "${CMPLR_ROOT:-}" != "/opt/intel/oneapi/compiler/$IE_ONEAPI_COMPILER" ]; then
  if [ -f /opt/intel/oneapi/setvars.sh ]; then
    _ie_cfg="$(mktemp)"
    printf 'compiler=%s\n' "$IE_ONEAPI_COMPILER" > "$_ie_cfg"
    # setvars.sh reads unset variables: a caller under `set -u` (the gate scripts) would die inside it.
    case $- in *u*) _ie_nounset=1; set +u ;; *) _ie_nounset= ;; esac
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh --force --config="$_ie_cfg" >/dev/null 2>&1
    [ -n "$_ie_nounset" ] && set -u
    rm -f "$_ie_cfg"
    unset _ie_cfg _ie_nounset
  fi
fi

# Project shortcuts
export IE_ROOT="$_IE_ROOT"
unset _IE_ROOT

# SYCL target seed for a FRESH configure (CMakeLists.txt reads it into the
# IE_SYCL_TARGET cache default; an existing build dir keeps its own value).
# spir64 = JIT (default; the driver cache + SYCL_CACHE_PERSISTENT make repeat
# loads free). intel_gpu_bmg_g31 = AOT for the B70, e.g.
#   IE_SYCL_TARGET=intel_gpu_bmg_g31 cmake -S . -B build-aot -G Ninja
export IE_SYCL_TARGET="${IE_SYCL_TARGET:-spir64}"

echo "ie env: cmake=$(command -v cmake), ninja=$(command -v ninja), icpx=$(command -v icpx)"
echo "ie env: SYCL target = $IE_SYCL_TARGET"
echo "ie env: \$IE_ROOT    = $IE_ROOT"
