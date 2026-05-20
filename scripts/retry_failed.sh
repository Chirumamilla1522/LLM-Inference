#!/usr/bin/env bash
# Re-run configs that failed (401 / stale repos / memory) with current repo map.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
source scripts/hf_env.sh

HARDWARE="${1:-Mac M3}"

if ! .venv/bin/huggingface-cli whoami &>/dev/null; then
  echo "Run: ./scripts/hf_login.sh  (needed for Mistral fp16)"
  exit 1
fi

echo "=== Failures in existing JSON ==="
python scripts/validate_results.py --hardware "$HARDWARE" --failures-only || true
echo ""

CONFIGS=(
  fp16 fp16+kv_cache fp16+prefill fp16+kv_cache+prefill
  w8 w8+kv_cache w8+prefill w8+kv_cache+prefill
)

for preset in llama3-8b mistral-7b; do
  for config in "${CONFIGS[@]}"; do
    echo ">>> $preset / $config"
    python scripts/run_benchmark.py --preset "$preset" --config "$config" --hardware "$HARDWARE"
  done
done

echo "Qwen on 24GB Mac (4-bit only):"
python scripts/run_benchmark.py --preset qwen-35b --config w4 --hardware "$HARDWARE" || true
