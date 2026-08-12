#!/usr/bin/env bash
# Regenerate Medium article figures into per-article folders.
#
# Usage:
#   ./scripts/regenerate_medium_images.sh                  # all articles
#   ./scripts/regenerate_medium_images.sh "Mac M3"          # all, custom HW label
#   ./scripts/regenerate_medium_images.sh "Mac M3" 00       # only Part 1
#   ./scripts/regenerate_medium_images.sh "Mac M3" 01-weight-quantization
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
HW="${1:-Mac M3}"
ARTICLE="${2:-}"

ART_ARGS=()
if [[ -n "$ARTICLE" ]]; then
  ART_ARGS=(--article "$ARTICLE")
  echo "Regenerating images for article: $ARTICLE"
else
  echo "Regenerating images for all articles"
fi

python scripts/plot_medium_diagrams.py "${ART_ARGS[@]}"
python scripts/plot_medium_charts.py --hardware "$HW" "${ART_ARGS[@]}"
python scripts/plot_medium_deep.py "${ART_ARGS[@]}"
python scripts/plot_paper_redraws.py "${ART_ARGS[@]}"
python scripts/organize_medium_images.py "${ART_ARGS[@]}"
python scripts/build_medium_publish.py "${ART_ARGS[@]}"

echo "Done. Per-article PNGs:"
if [[ -n "$ARTICLE" ]]; then
  # resolve short form via python
  SLUG="$(python -c "import sys; sys.path.insert(0,'scripts'); from medium_image_layout import resolve_article; print(resolve_article('$ARTICLE'))")"
  find "docs/medium/images/$SLUG" -name '*.png' | wc -l | tr -d ' '
  ls "docs/medium/images/$SLUG"
else
  find docs/medium/images -path '*/_source/*' -prune -o -name '*.png' -print | wc -l | tr -d ' '
  ls -d docs/medium/images/*/
fi
