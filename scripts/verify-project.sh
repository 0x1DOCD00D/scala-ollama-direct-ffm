#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "Project: $ROOT"
echo "Expected version: 0.3.0-direct-gguf"

grep -q 'case "--gguf"' src/main/scala/directollama/Cli.scala
grep -q 'options.gguf match' src/main/scala/directollama/Main.scala
grep -q '0.3.0-direct-gguf' src/main/scala/directollama/ProjectVersion.scala

echo "OK: --gguf parser is present"
echo "OK: direct GGUF resolution is present"
echo "OK: version marker is present"

echo
if command -v java >/dev/null 2>&1; then
  java -version 2>&1 | head -n 1
else
  echo "WARNING: java not found"
fi

if command -v cmake >/dev/null 2>&1; then
  cmake --version | head -n 1
else
  echo "WARNING: cmake not found; install with 'brew install cmake' on macOS"
fi
