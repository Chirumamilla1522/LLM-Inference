#!/usr/bin/env bash
# Run M3-vs-M5 headline articles on a Mac M5 Max machine.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
source scripts/hf_env.sh 2>/dev/null || true

HARDWARE="${1:-Mac M5 Max}"

echo "=== M5 ladder: Article 4 (model size) + Article 5 (capstone) ==="
echo "Hardware label: $HARDWARE"
echo ""

if ! huggingface-cli whoami &>/dev/null; then
  echo "Run ./scripts/hf_login.sh first."
  exit 1
fi

./scripts/run_article.sh 4 "$HARDWARE"
./scripts/run_article.sh 5 "$HARDWARE"

python scripts/generate_article_tables.py --hardware "$HARDWARE" --article 4 -o "docs/articles/_generated/tables_article04_M5.md"
python scripts/generate_article_tables.py --hardware "$HARDWARE" --article 5 -o "docs/articles/_generated/tables_article05_M5.md"
python scripts/report.py --hardware "$HARDWARE" -o "docs/articles/_generated/report_M5.md"

echo "Done. Compare with Mac M3: python scripts/generate_article_tables.py --hardware 'Mac M3'"
