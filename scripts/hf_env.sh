#!/usr/bin/env bash
# Load Hugging Face token from .env (if present) for mlx-lm downloads.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT}/.env"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

# huggingface_hub reads HF_TOKEN / HUGGING_FACE_HUB_TOKEN / ~/.cache/huggingface/token
export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
