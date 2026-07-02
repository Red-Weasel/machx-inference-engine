# Source this file to set up a build shell.
#   source scripts/env.sh
#
# Adds locally-installed cmake/ninja to PATH and initializes oneAPI 2026.0.

# Find the project root regardless of cwd.
_IE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

# Local tooling (cmake, ninja) installed under ~/.local/bin
case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) export PATH="$HOME/.local/bin:$PATH" ;;
esac

# oneAPI 2026.0 — DPC++ icpx + Level Zero + SYCL runtime
if [ -z "${ONEAPI_ROOT:-}" ]; then
  if [ -f /opt/intel/oneapi/setvars.sh ]; then
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
  fi
fi

# Project shortcuts
export IE_ROOT="$_IE_ROOT"
unset _IE_ROOT

# Pin the AOT target for B70 unless caller overrides
export IE_SYCL_TARGET="${IE_SYCL_TARGET:-intel_gpu_bmg_g31}"

echo "ie env: cmake=$(command -v cmake), ninja=$(command -v ninja), icpx=$(command -v icpx)"
echo "ie env: SYCL target = $IE_SYCL_TARGET"
echo "ie env: \$IE_ROOT    = $IE_ROOT"
