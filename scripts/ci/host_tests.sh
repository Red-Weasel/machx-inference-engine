#!/bin/bash
# The unit tests that need no GPU, no SYCL compiler and no model file: built with a plain C++20 compiler and run.
# This is what CI can run on any machine (.github/workflows/host-tests.yml); the GPU tests need an Arc card and are
# run by scripts' gates on the development box.
#   scripts/ci/host_tests.sh            env: CXX (default g++), JOBS (default 4), KEEP=1 keeps the build directory
# A test is listed as  name: extra sources...  -- the same sources tests/CMakeLists.txt gives it. When a new host-only test is
# added there, add it here; the check at the end fails if CMake has a host-only test this list does not.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT" || exit 2
CXX="${CXX:-g++}"; JOBS="${JOBS:-4}"
B="$(mktemp -d "${TMPDIR:-/tmp}/ie-host-tests.XXXXXX")"
TESTS=(
  "prompt_guard_test:"
  "glm5_memory_policy_test:"
  "glm5_lanes_test: src/model/glm5_lanes.cpp"
  "lanes_serve_test: src/engine/lanes_serve.cpp src/model/glm5_lanes.cpp"
  "q35m_lanes_test: src/engine/lanes_serve.cpp src/model/glm5_lanes.cpp"
  "q27_lanes_test: src/engine/lanes_serve.cpp src/model/glm5_lanes.cpp"
  "deep_pipe_test:"
  "lanes_auto_test:"
  "dn_ladder_test:"
  "stop_sequences_test:"
  "dequant_ref_test:"
  "prompt_cache_slots_test:"
  "hugetlb_pool_test:"
  "ds41_serve_rules_test:"
  "mimo26_host_rules_test: src/model/mimo26_host_rules.cpp"
  "mimo26_lanes_test: src/model/mimo26_lanes.cpp"
  "ds41_prefix_key_test: src/model/deepseek41_prefix_key.cpp"
  "idle_spin_test: src/server/idle_spin.cpp"
  "q4e_lane_ladder_test:"
)
build_one() {
  local name="${1%%:*}" extra="${1#*:}"
  # shellcheck disable=SC2086
  "$CXX" -std=c++20 -O1 -g0 -pthread -Iinclude -Isrc -Ithird_party -o "$B/$name" "tests/unit/$name.cpp" $extra > "$B/$name.build.log" 2>&1
  echo $? > "$B/$name.build.rc"
}
export -f build_one; export CXX B
printf '%s\n' "${TESTS[@]}" | xargs -P "$JOBS" -I{} bash -c 'build_one "{}"'
pass=0; fail=0; failed=()
for t in "${TESTS[@]}"; do
  name="${t%%:*}"
  if [ "$(cat "$B/$name.build.rc" 2>/dev/null)" != "0" ]; then
    echo "BUILD FAIL  $name: $(grep -m1 -E 'error' "$B/$name.build.log" | cut -c1-160)"; fail=$((fail + 1)); failed+=("$name"); continue
  fi
  if timeout 300 "$B/$name" > "$B/$name.log" 2>&1; then echo "PASS        $name"; pass=$((pass + 1))
  else echo "FAIL        $name (exit $?): $(tail -1 "$B/$name.log" | cut -c1-140)"; fail=$((fail + 1)); failed+=("$name"); fi
done
# the list against CMake: a test executable CMake links without ie_core / ie_sycl_flags must be in TESTS (or in SKIP below)
SKIP="supervisor_test"   # links the ie_supervisor library (the server's process code): built by the full build
missing=$(python3 - "$SKIP" "${TESTS[@]}" <<'PY'
import re, sys
skip = set(sys.argv[1].split()); have = {t.split(":")[0] for t in sys.argv[2:]}
s = open("tests/CMakeLists.txt").read()
exes = set(re.findall(r"add_executable\((\w+)", s)); links = {}
for m in re.finditer(r"target_link_libraries\((\w+)\s+(?:PRIVATE|PUBLIC)?\s*([^\)]+)\)", s): links[m[1]] = links.get(m[1], "") + " " + m[2]
host = {t for t in exes if not re.search(r"ie_core|ie_sycl|sycl|dnnl", links.get(t, ""))}
print(" ".join(sorted(host - have - skip)))
PY
)
[ -n "$missing" ] && { echo "NOT LISTED  host-only tests in tests/CMakeLists.txt missing from this script: $missing"; fail=$((fail + 1)); }
echo "host tests: $pass passed, $fail failed$( [ ${#failed[@]} -gt 0 ] && echo " (${failed[*]})" )  [$("$CXX" --version | head -1)]"
[ "${KEEP:-0}" = "1" ] && echo "build dir: $B" || rm -rf "$B"
[ $fail -eq 0 ]
