#!/usr/bin/env bash
# Complete Mac M3 benchmark pass for the article series.
#
# Usage:
#   ./scripts/run_m3_all.sh              # recommended (~2–8 h + downloads)
#   ./scripts/run_m3_all.sh full         # everything including Article 5 full sweep (~1–2 days)
#   ./scripts/run_m3_all.sh quick        # smoke: articles 0,2,10 only
#   ./scripts/run_m3_all.sh --from-checkpoint   # resume interrupted sweeps
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate
# shellcheck disable=SC1091
source scripts/hf_env.sh 2>/dev/null || true

HARDWARE="${HARDWARE:-Mac M3}"
MODE="${1:-standard}"
shift || true

EXTRA=("$@")
FROM_CKPT=()
if [[ " ${EXTRA[*]:-} " == *" --from-checkpoint "* ]]; then
  FROM_CKPT=(--from-checkpoint)
fi

echo "=============================================="
echo " LLM-Inference — Mac M3 full run"
echo " Hardware: $HARDWARE"
echo " Mode:     $MODE"
echo "=============================================="

if ! huggingface-cli whoami &>/dev/null; then
  echo "ERROR: Hugging Face login required."
  echo "  ./scripts/hf_login.sh"
  echo "  Accept licenses at huggingface.co for Llama 3.1, Mistral, etc."
  exit 1
fi

echo ""
echo ">>> HF repo check"
python scripts/run_benchmark.py --hf-check

run_article() {
  local id="$1"
  shift
  echo ""
  echo "=============================================="
  echo " Article $id"
  echo "=============================================="
  ./scripts/run_article.sh "$id" "$HARDWARE" "$@" "${FROM_CKPT[@]}"
}

case "$MODE" in
  quick)
    run_article 0
    run_article 2
    ./scripts/run_article.sh 10 "$HARDWARE" "${FROM_CKPT[@]}" || true
    for id in 8 9 11; do
      run_article "$id"
    done
    ;;
  standard)
    # Core MLX articles (M3 is the baseline machine in the series)
    for id in 0 1 2 3 4 6 7; do
      run_article "$id" "${FROM_CKPT[@]}"
    done
    # Article 5: hero fp16 vs optimized only (skip 224-run full sweep on 24GB)
    run_article 5 --runs-only
    # Article 10: MLX vs llama.cpp (+ optional server via compare flags)
    ./scripts/run_article.sh 10 "$HARDWARE" "${FROM_CKPT[@]}" || true
    # Concept manifests
    for id in 8 9 11; do
      run_article "$id"
    done
    ;;
  full)
    for id in 0 1 2 3 4 5 6 7; do
      run_article "$id" "${FROM_CKPT[@]}"
    done
    ./scripts/run_article.sh 10 "$HARDWARE" --with-server "${FROM_CKPT[@]}" 2>/dev/null \
      || ./scripts/run_article.sh 10 "$HARDWARE" "${FROM_CKPT[@]}" || true
    for id in 8 9 11; do
      run_article "$id"
    done
    ;;
  *)
    echo "Unknown mode: $MODE (use: standard | full | quick)"
    exit 1
    ;;
esac

echo ""
echo ">>> Validate results"
python scripts/validate_results.py --hardware "$HARDWARE" --failures-only || true

echo ""
echo ">>> Report + tables"
python scripts/report.py --hardware "$HARDWARE" \
  -o "docs/articles/_generated/report_Mac_M3.md" || true

if python -c "import matplotlib" 2>/dev/null; then
  python scripts/plot_results.py --hardware "$HARDWARE" --preset llama3-8b || true
fi

echo ""
echo "=============================================="
echo " M3 run complete"
echo " Results:  results/Mac_M3/"
echo " Report:   docs/articles/_generated/report_Mac_M3.md"
echo " Next:     run ./scripts/run_m5_ladder.sh on Mac M5 Max for M3 vs M5 charts"
echo "=============================================="
