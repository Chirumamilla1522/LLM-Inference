#!/usr/bin/env bash
# Hugging Face CLI helpers — prefer `hf` (huggingface-cli is deprecated).
# Source from other scripts: source "$(dirname "$0")/hf_cli.sh"

_hf_root() {
  if [[ -n "${ROOT:-}" ]]; then
    echo "$ROOT"
  else
    cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
  fi
}

hf_bin() {
  local root
  root="$(_hf_root)"
  if command -v hf &>/dev/null; then
    command -v hf
  elif [[ -x "${root}/.venv/bin/hf" ]]; then
    echo "${root}/.venv/bin/hf"
  elif [[ -x "${root}/.venv/bin/huggingface-cli" ]]; then
    echo "${root}/.venv/bin/huggingface-cli"
  else
    return 1
  fi
}

hf_is_legacy_cli() {
  [[ "$(hf_bin 2>/dev/null || echo "")" == *huggingface-cli* ]]
}

hf_whoami_ok() {
  local bin
  bin="$(hf_bin)" || return 1
  if hf_is_legacy_cli; then
    "$bin" whoami &>/dev/null
  else
    "$bin" auth whoami &>/dev/null
  fi
}

hf_login() {
  local bin
  bin="$(hf_bin)" || {
    echo "ERROR: Install huggingface_hub in venv: pip install huggingface_hub"
    return 1
  }
  if hf_is_legacy_cli; then
    "$bin" login
  else
    "$bin" auth login
  fi
}

hf_whoami() {
  local bin
  bin="$(hf_bin)" || return 1
  if hf_is_legacy_cli; then
    "$bin" whoami
  else
    "$bin" auth whoami
  fi
}
