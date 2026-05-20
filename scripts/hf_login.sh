#!/usr/bin/env bash
# Log in to Hugging Face (needed for gated fp16 models like Mistral).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate

echo "Log in to Hugging Face (token from https://huggingface.co/settings/tokens)"
echo "Also accept model licenses on huggingface.co for:"
echo "  - mlx-community/Mistral-7B-Instruct-v0.3-bf16"
echo ""

"${ROOT}/.venv/bin/huggingface-cli" login

echo ""
echo "Verify:"
"${ROOT}/.venv/bin/huggingface-cli" whoami
echo ""
echo "Optional: save token to .env for scripts:"
echo "  echo 'HF_TOKEN=hf_...' >> .env"
