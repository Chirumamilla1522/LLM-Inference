#!/usr/bin/env bash
# Log in to Hugging Face (needed for gated fp16 models like Mistral).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate
# shellcheck disable=SC1091
source "${ROOT}/scripts/hf_cli.sh"

echo "Log in to Hugging Face (token from https://huggingface.co/settings/tokens)"
echo "Uses: hf auth login  (huggingface-cli is deprecated)"
echo ""
echo "Also accept model licenses on huggingface.co for gated repos, e.g.:"
echo "  - mlx-community/Mistral-7B-Instruct-v0.3-bf16"
echo "  - mlx-community/Meta-Llama-3.1-8B-Instruct-bf16"
echo ""

hf_login

echo ""
echo "Verify:"
hf_whoami
echo ""
echo "Optional: save token to .env for scripts:"
echo "  echo 'HF_TOKEN=hf_...' >> .env"
echo "  (scripts load .env via scripts/hf_env.sh)"
