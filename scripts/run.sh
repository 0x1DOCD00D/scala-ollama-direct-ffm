#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${JAVA_HOME:-}" != "" ]]; then
  export PATH="$JAVA_HOME/bin:$PATH"
fi

exec sbt "run $*"
