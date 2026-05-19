#!/usr/bin/env bash
# Full sweep tuned for Apple Silicon (fp16 / w8 / w4 / w2 × runtime opts).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate

HARDWARE="${1:-Mac M3}"
RAM_GB="$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f", $1/1024/1024/1024}' || echo 24)"

EXTRA=()
if [[ "${RAM_GB}" -lt 36 ]]; then
  echo "Detected ~${RAM_GB}GB RAM — skipping qwen-35b (needs 36GB+)."
  echo "On M5 Max / 64GB+ Mac: ./scripts/run_full_sweep.sh \"Mac M5 Max\" --include-qwen"
  EXTRA+=(--skip-presets qwen-35b)
fi

python scripts/run_benchmark.py \
  --sweep \
  --all-models \
  --hardware "$HARDWARE" \
  "${EXTRA[@]}" \
  -n 3 \
  -p 512 \
  -g 128
