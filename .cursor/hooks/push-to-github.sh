#!/bin/bash
# After each agent turn completes, commit workspace changes and push to GitHub.
# Disable: export CURSOR_AUTO_PUSH=0
set -euo pipefail

if [[ "${CURSOR_AUTO_PUSH:-1}" == "0" ]]; then
  exit 0
fi

ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
cd "$ROOT"

LOG_DIR=".cursor/hooks/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/push.log"

log() {
  printf '[%s] %s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$*" >>"$LOG_FILE"
}

# Avoid overlapping runs if stop fires twice in quick succession.
LOCK_FILE=".cursor/hooks/.push.lock"
if ! mkdir "$LOCK_FILE" 2>/dev/null; then
  log "Skipped: another push hook is already running"
  exit 0
fi
trap 'rmdir "$LOCK_FILE" 2>/dev/null || true' EXIT

SENSITIVE_PATTERNS=(
  .env
  .env.*
  credentials.json
  '*.pem'
  id_rsa
  id_ed25519
)

has_sensitive_changes() {
  local pattern path
  for pattern in "${SENSITIVE_PATTERNS[@]}"; do
    while IFS= read -r path; do
      [[ -n "$path" ]] || continue
      log "Blocked: refusing to commit sensitive path: $path"
      return 0
    done < <(git status --porcelain -- "$pattern" 2>/dev/null || true)
  done
  return 1
}

if has_sensitive_changes; then
  exit 0
fi

if git diff --quiet && git diff --cached --quiet; then
  if [[ -z "$(git ls-files --others --exclude-standard)" ]]; then
    log "No changes to commit"
    exit 0
  fi
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

git add -A

for pattern in "${SENSITIVE_PATTERNS[@]}"; do
  git reset -q HEAD -- $pattern 2>/dev/null || true
done

if git diff --cached --quiet; then
  log "Nothing staged after excluding sensitive files"
  exit 0
fi

MSG="chore(agent): sync Cursor agent changes (${TIMESTAMP})"

if ! git commit -m "$MSG"; then
  log "Commit failed on branch ${BRANCH}"
  exit 0
fi

if git push origin "$BRANCH" >>"$LOG_FILE" 2>&1; then
  log "Committed and pushed to origin/${BRANCH}"
else
  log "Commit succeeded but push failed on origin/${BRANCH} (see log for details)"
fi

exit 0
