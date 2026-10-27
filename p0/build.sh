#!/usr/bin/env bash
# Build the P0 kernels into p0/p0kit/_lib/libp0.so (run from anywhere).
# Override the compiler with CXX=..., the flags with CXXFLAGS=...
set -euo pipefail
cd "$(dirname "$0")"
CXX="${CXX:-g++}"
FLAGS="${CXXFLAGS:--O3 -march=native}"
mkdir -p p0kit/_lib
echo "compiler: $("$CXX" --version | head -1)"
"$CXX" -std=c++17 $FLAGS -fPIC -shared -Wall -Wextra -Wno-unused-parameter \
    -o p0kit/_lib/libp0.so cpp/p0_dyn.cpp cpp/p0_krmax.cpp cpp/p0_bnb.cpp cpp/p0_graph.cpp
echo "built p0/p0kit/_lib/libp0.so with: $CXX -std=c++17 $FLAGS"
