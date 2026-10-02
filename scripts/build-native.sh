#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$ROOT/external/llama.cpp}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-master}"
BUILD_DIR="$ROOT/native/build"

for tool in cmake git; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "ERROR: $tool is required but was not found on PATH." >&2
    if [[ "$tool" == "cmake" && "$(uname -s)" == "Darwin" ]]; then
      echo "Install it with: brew install cmake" >&2
    fi
    exit 1
  fi
done

if [[ "$(uname -s)" == "Darwin" ]] && ! xcrun --find clang++ >/dev/null 2>&1; then
  echo "ERROR: Apple Command Line Tools are required." >&2
  echo "Install them with: xcode-select --install" >&2
  exit 1
fi

if [[ ! -f "$LLAMA_CPP_DIR/include/llama.h" ]]; then
  echo "llama.cpp not found at $LLAMA_CPP_DIR"
  echo "Cloning $LLAMA_CPP_REF..."
  mkdir -p "$(dirname "$LLAMA_CPP_DIR")"
  git clone --depth 1 --branch "$LLAMA_CPP_REF" https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
fi

CMAKE_ARGS=(
  -S "$ROOT/native"
  -B "$BUILD_DIR"
  -DLLAMA_CPP_DIR="$LLAMA_CPP_DIR"
  -DCMAKE_BUILD_TYPE=Release
)

if [[ "${GGML_CUDA:-0}" == "1" ]]; then
  CMAKE_ARGS+=( -DGGML_CUDA=ON )
fi

if [[ "${GGML_METAL:-}" == "0" ]]; then
  CMAKE_ARGS+=( -DGGML_METAL=OFF )
fi

cmake "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" --config Release -j "${BUILD_JOBS:-8}"

echo
echo "Built native library under $BUILD_DIR/ffi"
find "$BUILD_DIR/ffi" -maxdepth 1 -type f -print
