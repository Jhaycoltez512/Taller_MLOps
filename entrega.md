# Entrega — MLOps · Defiende el servidor

## Equipo y repositorios

- **Integrante y grupo:** Mateo Hoyos Cárdenas — Grupo A (entrega individual).
- **Repositorio de código:** https://github.com/Jhaycoltez512/Taller_MLOps (rama `main`).
- **Commit final de la entrega:** el commit de `main` que contiene esta versión de `entrega.md`;
  su SHA completo se indica en el mensaje de entrega (un archivo no puede contener el SHA de su
  propio commit).
- **Repositorio de modelos:** https://huggingface.co/yeico512/unsw-nb15-defiende-el-servidor (público).
- **Almacenamiento DVC:** https://dagshub.com/Jhaycoltez512/Taller_MLOps.dvc (repositorio público
  en DagsHub; ver Acceso).

## Versiones

| Campo | v1 | v2 |
|---|---|---|
| Configuración | `configs/experiments/exp01_logreg_base.yaml` | `configs/experiments/exp04_hist_gradient_boosting.yaml` |
| Commit de código que entrenó el modelo | `fecc3296fb838bbc6ec8296959021f9fbeb70cc4` | `fecc3296fb838bbc6ec8296959021f9fbeb70cc4` |
| Commit de código que identifica la versión | `b15a8b2185be872cafb8b37041074a22ece1063e` (publica v1) | `adcd9553e061cbeceda7d0c3fa5bf93a5f6ba34c` (selecciona y congela v2) |
| Datos DVC (original / preparados) | `23b017bf013ca3cb801267d722c28c7a.dir` / `e2f660e3d42e641a595c423b0d2af0ed.dir` | igual que v1 |
| Commit Git de esas referencias DVC | `e0a5e28569a06ebdaea6bb3eacc0695031dde736` | igual que v1 |
| Run ID de MLflow | `b1c856bd8de8457a94df5e64d12c0d58` | `98001a529d484e79a7f6b0cf95dd962d` |
| Revisión de Hugging Face (SHA completo) | `896ee3984613e60d2af1c5297163524216fec34b` (tag `v1`) | `424eef599abdce7630cfc369505487755b9763f4` (tag `v2`) |
| SHA-256 del modelo (`model.skops`) | `feb7d024b424fc072332e603ac394024bd7f297f2a2b2ab68b5630a2503d1771` | `562ba1bca9c762c4789a7dcdce4064508d7dc63b180e40b00e57eb302503489c` |
| Umbral (ataque si `score_attack >= umbral`) | 0.773 | 0.639 |
| Validación: recall / precision / FPR / macro-F1 | 0.896 / 0.972 / 5.44 % / 0.902 | 0.956 / 0.976 / 4.99 % / 0.948 |

Las dos versiones se entrenaron con el mismo código (`fecc329`) y difieren en la configuración:
modelo (regresión logística → HistGradientBoosting), preprocesamiento (one-hot + estandarización →
ordinal) y umbral (0.773 → 0.639). El model card de v2 recibió el resultado de test en la revisión
`dd7f3fed3214b7a7d39b20fe067bacb926904c36`, que solo cambia `README.md`; la revisión evaluada sigue
siendo `424eef59…` y el modelo conserva su SHA-256.

En Git, v1 y v2 se identifican por los commits `b15a8b2` y `adcd955` (el enunciado acepta
commits o tags). Opcionalmente se pueden añadir tags `v1`/`v2` sobre esos commits con los
comandos de la sección final.

## Evidencias

| Requisito | Archivo |
|---|---|
| Auditoría: fuentes, SHA-256, filas/columnas, clases, faltantes, duplicados, contradicciones, política, variables, semilla, splits y cruces | `evidencias/auditoria.json` |
| Tabla de experimentos (ID, hipótesis, familia, parámetros, semilla, commit, DVC, run ID, umbral, métricas, conclusión) | `evidencias/experimentos.csv` |
| Detección por categoría en validación | `evidencias/errores_por_categoria.csv` |
| Exportación de MLflow (4 runs: parámetros, métricas, tags y artefactos) | `evidencias/mlflow/` (`runs.json` y una carpeta por run ID) |
| Selección de v2 congelada antes del test | `evidencias/seleccion.json` (commit `adcd955`) |
| Publicación y verificación de cada versión en Hugging Face | `evidencias/versiones/v1.json`, `evidencias/versiones/v2.json` |
| Métricas de test (versión, hash del modelo, hash del dataset, filas, umbral, clase positiva, matriz de confusión) | `evidencias/metricas-test.json` |
| Predicciones de test (una fila por registro: `row_id`, `y_true`, `score_attack`, `y_pred`) | `evidencias/predicciones-test.csv` |
| Análisis del resultado de test | `evidencias/analisis-test.json` |
| Prueba de recuperación (entradas, salidas antes y después, hashes, comandos, log y comparación) | `evidencias/recuperacion/` |
| Informe | `informe/informe.pdf` |

`row_id` es `test:<id>` (o `train:<id>` en validación), donde `id` es el número de fila (1..N) del
CSV oficial. Regla de decisión: `y_pred = 1` si `score_attack >= umbral`, con
`score_attack = predict_proba[:, 1]`.

## Acceso

- **GitHub:** repositorio público; no requiere invitación.
- **DagsHub (datos):** repositorio público; DagsHub exige un token incluso para leer. Con cualquier
  cuenta de DagsHub: `dvc remote modify --local origin user <usuario>` y
  `dvc remote modify --local origin password <token>`.
- **Hugging Face:** repositorio público; la descarga no requiere cuenta.
- No se comparten tokens: las credenciales van en `.dvc/config.local` o variables de entorno.

## Comandos

```bash
# Instalación (Python 3.13)
git clone https://github.com/Jhaycoltez512/Taller_MLOps.git && cd Taller_MLOps
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt

# Datos
export DAGSHUB_USER=<usuario> DAGSHUB_TOKEN=<token>
dvc remote modify --local origin user "$DAGSHUB_USER"
dvc remote modify --local origin password "$DAGSHUB_TOKEN"
dvc pull && dvc status

# Reproducir un experimento (MLflow aparte para no mezclar runs)
MLFLOW_TRACKING_URI=sqlite:///repro.db python -m src.models.train \
  --config configs/experiments/exp04_hist_gradient_boosting.yaml

# Descargar v2 por SHA y predecir
python -m src.models.predict --repo-id yeico512/unsw-nb15-defiende-el-servidor \
  --revision 424eef599abdce7630cfc369505487755b9763f4 \
  --input evidencias/recuperacion/entrada.csv --output preds_v2.csv

# Recalcular métricas de test y repetir la prueba de recuperación
python -m src.models.recompute_metrics
bash scripts/recuperacion.sh /tmp/recuperacion-nueva 91f71a216abc7802ca92faf7b9ac22c317e11b10
```

## Tiempo y recursos medidos

Contenedor Linux x86-64, 4 CPU (Intel Xeon 2.80 GHz), 15 GB de RAM, sin GPU.

| Paso | Tiempo |
|---|---|
| Instalar `requirements.lock.txt` en un entorno nuevo | 298 s |
| `dvc pull` desde DagsHub (47 MB originales + preparados) | 10 s |
| `dvc repro` (preparación y auditoría) | ~150 s |
| Experimentos (ajuste + predicciones out-of-fold para el umbral) | exp01 31 s · exp02 31 s · exp03 37 s · exp04 16 s |
| Prueba de recuperación completa | 327 s |

## Limitaciones conocidas

- UNSW-NB15 es un dataset sintético de 2015: los resultados no demuestran eficacia sobre una red actual.
- La FPR de v2 sube de 5,0 % en validación a 18,2 % en test: en test el 51,8 % del tráfico normal
  tiene TTL típico de los atacantes del banco de pruebas (23 % en train/val) y el modelo usa ese
  artefacto. Se conserva el resultado sin ajustar modelo ni umbral.
- El 8,1 % de las filas de test comparte vector de entradas con train; se reporta y no se eliminan.
- Los `Fuzzers` son la principal fuente de ataques no detectados (~26 % sin detectar en validación).
- Seguimiento de MLflow local: la evidencia consultable es la exportación en `evidencias/mlflow/`.
- El modelo de exp03 (52 MB) no se incluye en la exportación; queda su SHA-256.

## Uso de IA, código externo y contribuciones

- **Asistente de IA:** Claude Code (Anthropic) se usó para escribir el código (datos,
  entrenamiento, publicación, evaluación y recuperación), redactar la documentación y analizar
  resultados. El integrante revisó y ejecutó las verificaciones indicadas abajo.
  Verificaciones automáticas incluidas: métricas recalculadas con scikit-learn, hashes de datos y
  modelos, reproducción de un experimento en un clon aparte y prueba de recuperación en entorno nuevo.
- **Código externo:** no se copió código de terceros; se usan las bibliotecas listadas en
  `requirements.lock.txt` (scikit-learn, pandas, DVC, MLflow, huggingface_hub, skops, entre otras)
  bajo sus licencias.
- **Datos:** UNSW-NB15 (Moustafa y Slay, 2015), descargado de la fuente oficial de UNSW.
- **Contribución:** trabajo individual de Mateo Hoyos Cárdenas.
  - **Desarrollo:** configuración del repositorio, las cuentas (GitHub, DagsHub, Hugging Face) y
    el entorno; descarga y versionado de los datos con DVC; auditoría, política de limpieza y
    diseño de las particiones; diseño de los cuatro experimentos y sus hipótesis; regla del umbral
    y regla de selección de v2; publicación de v1 y v2 en Hugging Face; interpretación de errores
    y del resultado de test (artefacto de TTL); prueba de recuperación; informe.
  - **Verificaciones:** SHA-256 de los CSV frente al anexo; `dvc pull` en un clon limpio con
    huellas iguales a la auditoría; `experimentos.csv` frente a la exportación de MLflow (run IDs,
    parámetros, métricas); orden de los commits (selección congelada antes del test); recálculo
    de las métricas de test con `python -m src.models.recompute_metrics`; descarga de v1 y v2 por
    SHA y coincidencia de predicciones en la prueba de recuperación; cifras del informe frente a
    `evidencias/`; ausencia de tokens en el repositorio y en el ZIP.

## Antes de enviar

```bash
# Opcional: tags de las versiones (desde un computador con permiso de push)
git fetch origin
git tag -a v1 b15a8b2185be872cafb8b37041074a22ece1063e -m "v1: exp01_logreg_base"
git tag -a v2 adcd9553e061cbeceda7d0c3fa5bf93a5f6ba34c -m "v2: exp04_hist_gradient_boosting"
git push origin v1 v2
```

Asunto del mensaje: `MLOps — Defiende el servidor — Grupo A — Mateo Hoyos Cárdenas`.
