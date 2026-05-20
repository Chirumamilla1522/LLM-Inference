#!/usr/bin/env bash
# Full sweep: all model sizes (tiny → XL), fp16/w8/w4/w2 × runtime opts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate
# shellcheck disable=SC1091
source scripts/hf_env.sh

HARDWARE="${1:-Mac M3}"
RAM_GB="$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f", $1/1024/1024/1024}' || echo 24)"

EXTRA=()
if [[ "${RAM_GB}" -lt 36 ]]; then
  echo "Detected ~${RAM_GB}GB RAM — skipping large presets (12B–72B, see: python scripts/list_models.py)."
  echo "On 64GB+ Mac: ./scripts/run_full_sweep.sh \"Mac M5 Max\" --include-large"
else
  EXTRA+=(--include-large)
fi

# Allow: ./scripts/run_full_sweep.sh "Mac M5 Max" --include-large
for arg in "${@:2}"; do
  EXTRA+=("$arg")
done

if ! .venv/bin/python scripts/run_benchmark.py --hf-check; then
  echo "Fix repo access above, then re-run."
  exit 1
fi

python scripts/run_benchmark.py \
  --sweep \
  --all-models \
  --hardware "$HARDWARE" \
  "${EXTRA[@]}" \
  -n 3 \
  -p 512 \
  -g 128
