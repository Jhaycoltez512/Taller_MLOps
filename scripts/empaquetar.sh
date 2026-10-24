#!/usr/bin/env bash
# Crea el ZIP de la entrega a partir de un commit (por defecto HEAD) con `git archive`:
# solo archivos versionados en Git, así que no incluye entornos virtuales, cachés, datos
# (data/original y data/prepared se recuperan con DVC), modelos locales ni credenciales.
#
# Uso: bash scripts/empaquetar.sh [commit]   ->  dist/proyecto-<commit corto>.zip
set -euo pipefail
REF="${1:-HEAD}"
COMMIT="$(git rev-parse "$REF")"
mkdir -p dist
OUT="dist/proyecto-${COMMIT:0:7}.zip"
git archive --format=zip --prefix=proyecto/ -o "$OUT" "$COMMIT"
echo "$OUT ($(du -h "$OUT" | cut -f1)) desde el commit $COMMIT"
sha256sum "$OUT"
