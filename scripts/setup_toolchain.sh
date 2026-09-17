#!/usr/bin/env bash
# scripts/setup_toolchain.sh — verify the build environment matches the pinned baseline.
# Pinned versions per PLAN.md Appendix A (research/01_hardware.md §5).
# Exits 0 when everything is at-or-above the floor; non-zero with a fix-it message otherwise.

set -euo pipefail

# Pinned floor versions
PIN_KERNEL_MIN="6.14"
PIN_OPENCL_MIN="26.05.37020.3"        # intel-opencl-icd
PIN_IGC_MIN="2.28.4"                  # intel-igc-core / opencl
PIN_ONEAPI_MIN="2026.0"               # /opt/intel/oneapi/<ver>
PIN_CMAKE_MIN="3.28"
PIN_NINJA_MIN="1.11"

red()    { printf "\033[31m%s\033[0m\n" "$*"; }
green()  { printf "\033[32m%s\033[0m\n" "$*"; }
yellow() { printf "\033[33m%s\033[0m\n" "$*"; }

vge() {
  # vge "$have" "$want" -> 0 if have >= want lex-numerically, 1 otherwise
  printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

probe() {
  local label="$1"; local got="$2"; local want="$3"
  if [ -z "$got" ]; then
    red   "  [MISS] $label: not found (need ≥ $want)"; FAIL=1
  elif vge "$got" "$want"; then
    green "  [ OK ] $label: $got (≥ $want)"
  else
    red   "  [LOW ] $label: $got (need ≥ $want)"; FAIL=1
  fi
}

FAIL=0
echo "=== Toolchain probe — pins from PLAN.md Appendix A ==="

# 1. Kernel
probe "Linux kernel" "$(uname -r | cut -d- -f1)" "$PIN_KERNEL_MIN"

# 2. GPU presence
if lspci -nn 2>/dev/null | grep -q '8086:e223'; then
  green "  [ OK ] Arc Pro B70 (PCI 8086:e223) detected"
else
  yellow "  [WARN] Arc Pro B70 (PCI 8086:e223) NOT detected — proceeding anyway"
fi

# 3. compute-runtime / OpenCL ICD
ocl_ver="$(dpkg-query -W -f='${Version}' intel-opencl-icd 2>/dev/null | head -1 | awk -F'-' '{print $1}')"
probe "intel-opencl-icd" "$ocl_ver" "$PIN_OPENCL_MIN"

# 4. IGC
igc_ver="$(dpkg-query -W -f='${Version}' intel-igc-core-2 2>/dev/null | head -1 | awk -F'-' '{print $1}')"
[ -z "$igc_ver" ] && igc_ver="$(dpkg-query -W -f='${Version}' intel-igc-opencl-2 2>/dev/null | head -1 | awk -F'-' '{print $1}')"
probe "intel-igc" "$igc_ver" "$PIN_IGC_MIN"

# 5. oneAPI
if [ -d /opt/intel/oneapi/compiler ]; then
  oneapi_ver="$(ls -1 /opt/intel/oneapi/compiler 2>/dev/null | grep -E '^[0-9]+\.[0-9]+' | sort -V | tail -1)"
  probe "oneAPI compiler" "$oneapi_ver" "$PIN_ONEAPI_MIN"
else
  red "  [MISS] /opt/intel/oneapi not found"; FAIL=1
fi

# 6. cmake / ninja
cm_ver="$(cmake --version 2>/dev/null | head -1 | awk '{print $3}')"
nj_ver="$(ninja --version 2>/dev/null)"
probe "cmake" "$cm_ver" "$PIN_CMAKE_MIN"
probe "ninja" "$nj_ver" "$PIN_NINJA_MIN"

# 7. SYCL device discovery (the load-bearing one)
if command -v sycl-ls >/dev/null 2>&1; then
  if sycl-ls 2>/dev/null | grep -q 'intel_gpu_bmg_g31\|0xe223'; then
    green "  [ OK ] sycl-ls reports intel_gpu_bmg_g31 / [0xe223]"
  else
    red   "  [FAIL] sycl-ls did not find the B70. Output:"
    sycl-ls 2>&1 | sed 's/^/         /'
    FAIL=1
  fi
else
  red "  [MISS] sycl-ls not on PATH (did you 'source scripts/env.sh'?)"; FAIL=1
fi

if [ "$FAIL" -ne 0 ]; then
  echo
  red "Toolchain probe FAILED. Address items above before running cmake."
  echo "Hint: 'source scripts/env.sh' from the project root first."
  exit 1
fi

echo
green "Toolchain ready."
