#!/usr/bin/env bash
# Article 10 — MLX vs llama.cpp comparison wrapper
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
HARDWARE="${1:-Mac M3}"
shift || true
exec python scripts/compare_runtimes.py --hardware "$HARDWARE" "$@"
