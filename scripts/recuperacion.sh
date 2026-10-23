#!/usr/bin/env bash
# Prueba de recuperación en una carpeta y un entorno nuevos, sin cachés de la ejecución original.
#
#   1. Clona el repositorio de código y fija el commit indicado; instala requirements.lock.txt
#      en un entorno virtual nuevo (sin caché de pip).
#   2. Recupera con DVC, desde DagsHub, los datos originales y preparados; verifica sus huellas.
#   3-4. Descarga v1 y v2 desde Hugging Face por su SHA completo (caché vacía) y predice sobre
#      las 30 filas fijas de evidencias/recuperacion/entrada.csv.
#   5. Compara con las salidas guardadas de los runs originales (tolerancia 1e-9 en el score).
#
# Uso:
#   export DAGSHUB_USER=<usuario> DAGSHUB_TOKEN=<token>      # acceso de lectura al remote DVC
#   bash scripts/recuperacion.sh <carpeta-nueva> [commit]
# Resultado: <carpeta-nueva>/resultado/ (log.txt, recuperado_v*.csv, comparacion.json, ...).
set -euo pipefail

WORK="${1:?uso: bash scripts/recuperacion.sh <carpeta-nueva> [commit]}"
REF="${2:-claude/nice-tesla-63i77h}"
REPO_URL="https://github.com/Jhaycoltez512/Taller_MLOps.git"
HF_REPO="yeico512/unsw-nb15-defiende-el-servidor"
: "${DAGSHUB_USER:?defina DAGSHUB_USER}" "${DAGSHUB_TOKEN:?defina DAGSHUB_TOKEN}"
DAGSHUB_USER="${DAGSHUB_USER#@}"

if [ -e "$WORK" ]; then echo "La carpeta $WORK ya existe: use una carpeta nueva." >&2; exit 1; fi
mkdir -p "$WORK/resultado"
WORK="$(cd "$WORK" && pwd)"
OUT="$WORK/resultado"
exec > >(tee "$OUT/log.txt") 2>&1

step() { echo; echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] $* ==="; }
START=$SECONDS

step "0. Entorno"
uname -srm; python3 --version; git --version
echo "carpeta: $WORK"

step "1. Código: clonar y fijar la revisión $REF"
git clone --quiet "$REPO_URL" "$WORK/repo"
cd "$WORK/repo"
git checkout --quiet "$REF"
COMMIT="$(git rev-parse HEAD)"
echo "commit: $COMMIT"
python3 -m venv "$WORK/venv"
# shellcheck disable=SC1091
source "$WORK/venv/bin/activate"
T=$SECONDS
pip install --quiet --no-cache-dir --upgrade pip
pip install --quiet --no-cache-dir -r requirements.lock.txt
echo "dependencias instaladas en $((SECONDS - T)) s"
python --version; dvc --version
pip freeze > "$OUT/pip-freeze.txt"

step "2. Datos: dvc pull desde el almacenamiento entregado (caché DVC vacía)"
test ! -d .dvc/cache && echo "caché DVC inicial: vacía"
dvc remote list
dvc remote modify --local origin user "$DAGSHUB_USER"
dvc remote modify --local origin password "$DAGSHUB_TOKEN"
T=$SECONDS
dvc pull
echo "datos recuperados en $((SECONDS - T)) s"
dvc status
sha256sum data/original/*.csv data/prepared/*.parquet data/prepared/split_assignment.csv | tee "$OUT/hashes-datos.txt"
python -m src.models.recovery verify-data

step "3-4. Modelos: descarga por SHA completo y predicción sobre la muestra fija"
export HF_HOME="$WORK/hf-home"   # caché de Hugging Face nueva y vacía
for V in v1 v2; do
  REV="$(python -c "import json; print(json.load(open('evidencias/versiones/$V.json'))['hf_revision'])")"
  echo "$V -> $HF_REPO@$REV"
  python -m src.models.predict --repo-id "$HF_REPO" --revision "$REV" --cache-dir "$WORK/hf-cache" \
    --input evidencias/recuperacion/entrada.csv --output "$OUT/recuperado_$V.csv"
done
find "$WORK/hf-cache" -name model.skops -exec sha256sum {} \; | tee "$OUT/hashes-modelos.txt"

step "5. Comparación con las salidas guardadas"
cp evidencias/recuperacion/entrada.csv evidencias/recuperacion/esperado_v1.csv \
   evidencias/recuperacion/esperado_v2.csv "$OUT/"
sha256sum "$OUT"/entrada.csv "$OUT"/esperado_v*.csv "$OUT"/recuperado_v*.csv | tee "$OUT/hashes-salidas.txt"
python -m src.models.recovery check --outdir "$OUT"

step "Fin: commit $COMMIT, $((SECONDS - START)) s en total"
