#!/bin/bash
# Configure and build into ./build (or $CG_BUILD_DIR). Extra CMake arguments
# come from $CG_CMAKE_ARGS (set by the env scripts) and from the command line:
#   bash scripts/build.sh -DCMAKE_CUDA_ARCHITECTURES=80
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC2086
cmake -S . -B "${CG_BUILD_DIR:-build}" ${CG_CMAKE_ARGS:-} "$@"
cmake --build "${CG_BUILD_DIR:-build}" -j "${CG_BUILD_JOBS:-8}"
