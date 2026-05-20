#!/usr/bin/env bash
# Run one article's benchmarks: ./scripts/run_article.sh 1 "Mac M3"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate
# shellcheck disable=SC1091
source scripts/hf_env.sh 2>/dev/null || true

ARTICLE="${1:-}"
HARDWARE="${2:-Mac M3}"

if [[ -z "$ARTICLE" ]]; then
  echo "Usage: $0 <article-id 0-11> [hardware-label]"
  echo "       $0 all [hardware-label]   # all benchmarked articles"
  python scripts/run_article.py --list
  exit 1
fi

if [[ "$ARTICLE" == "all" ]]; then
  exec python scripts/run_article.py --all-benchmarked --hardware "$HARDWARE" "${@:3}"
fi

exec python scripts/run_article.py --article "$ARTICLE" --hardware "$HARDWARE" "${@:3}"
