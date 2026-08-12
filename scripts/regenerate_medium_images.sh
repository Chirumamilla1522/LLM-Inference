#!/usr/bin/env bash
# Regenerate all Medium article figures (workflows + charts + deep plots).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
HW="${1:-Mac M3}"

python scripts/plot_medium_diagrams.py
python scripts/plot_medium_charts.py --hardware "$HW"
python scripts/plot_medium_deep.py
python scripts/plot_paper_redraws.py

echo "Done. Images under docs/medium/images/ ($(find docs/medium/images -name '*.png' | wc -l | tr -d ' ') PNGs)"
