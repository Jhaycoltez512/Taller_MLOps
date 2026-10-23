#!/usr/bin/env bash
# SessionStart: fija la autoría de los commits de este repo, también en sesiones
# de Claude Code en la nube/web (que traen su propia identidad de git).
set -euo pipefail

NAME="Mateo Hoyos"
EMAIL="132465005+Jhaycoltez512@users.noreply.github.com"

cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}"
git config user.name "$NAME"
git config user.email "$EMAIL"

# Las variables GIT_AUTHOR_*/GIT_COMMITTER_* tienen prioridad sobre git config,
# así que se exportan para cada comando de la sesión.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  cat >> "$CLAUDE_ENV_FILE" <<EOF
export GIT_AUTHOR_NAME="$NAME"
export GIT_AUTHOR_EMAIL="$EMAIL"
export GIT_COMMITTER_NAME="$NAME"
export GIT_COMMITTER_EMAIL="$EMAIL"
EOF
fi
