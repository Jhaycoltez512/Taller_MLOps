# Defiende el servidor — UNSW-NB15

Taller evaluable de MLOps (Universidad Externado de Colombia, 2026-II): clasificación binaria de
conexiones de red (`label`: 0 = normal, 1 = ataque, clase positiva) con los splits oficiales de
[UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset), versionando código (Git/GitHub),
datos (DVC + DagsHub), experimentos (MLflow) y modelos (Hugging Face Hub).

## Estado

- [x] Datos originales versionados con DVC (`data/original.dvc`) y almacenados en DagsHub
- [x] Auditoría, limpieza y particiones fijas (`dvc.yaml` → `evidencias/auditoria.json`)
- [x] Cuatro configuraciones de dos familias en MLflow (`evidencias/experimentos.csv`)
- [x] Modelos v1/v2 en Hugging Face Hub (`evidencias/versiones/`)
- [x] Selección congelada y evaluación única en test (`evidencias/seleccion.json`, `evidencias/metricas-test.json`)
- [x] Prueba de recuperación en entorno nuevo (`evidencias/recuperacion/`)
- [x] Índice de la entrega (`entrega.md`) e informe (`informe/informe.pdf`)

| Pieza | Dónde |
|---|---|
| Código | https://github.com/Jhaycoltez512/Taller_MLOps |
| Datos (DVC) | https://dagshub.com/Jhaycoltez512/Taller_MLOps (remote `https://dagshub.com/Jhaycoltez512/Taller_MLOps.dvc`) |
| Modelos | https://huggingface.co/yeico512/unsw-nb15-defiende-el-servidor (v1 y v2 por SHA) |
| Índice de evidencias | [`entrega.md`](entrega.md) |

## Requisitos

- Python 3.13 (probado con 3.13.16 en Linux x86-64, 4 CPU, 15 GB de RAM; no requiere GPU).
- Dependencias fijadas en `requirements.lock.txt` (`requirements.txt` lista solo las directas).
- Git. Para recuperar los datos, una cuenta de DagsHub (el repositorio es público, pero DagsHub
  pide un token incluso para leer). Los modelos se descargan sin cuenta.

## Instalación

```bash
git clone https://github.com/Jhaycoltez512/Taller_MLOps.git
cd Taller_MLOps
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt
```

En Debian/Ubuntu, instale dentro del entorno virtual: con el `pip` del sistema la
dependencia `antlr4-python3-runtime` de DVC no compila. La instalación tarda unos 5 minutos.

## Recuperar los datos (DVC + DagsHub)

El almacenamiento DVC es `https://dagshub.com/Jhaycoltez512/Taller_MLOps.dvc` (configurado en
`.dvc/config`). Con un usuario de DagsHub que tenga acceso de lectura al repositorio:

```bash
export DAGSHUB_USER=<usuario-dagshub> DAGSHUB_TOKEN=<token>   # DagsHub → User settings → Tokens
dvc remote modify --local origin user "$DAGSHUB_USER"
dvc remote modify --local origin password "$DAGSHUB_TOKEN"
dvc pull                                                     # data/original y data/prepared
dvc status                                                   # "Data and pipelines are up to date."
```

Las credenciales quedan en `.dvc/config.local`, que Git ignora. Para regenerar los datos
preparados desde los originales: `dvc repro` (determinista; `dvc.lock` fija los hashes).

### Versiones de datos

| Versión | Contenido | Referencia |
|---|---|---|
| Original | `data/original/`: `UNSW_NB15_training-set.csv` (175.341 filas), `UNSW_NB15_testing-set.csv` (82.332) y `MANIFEST.json` (fuente, fecha, SHA-256) | `data/original.dvc` |
| Preparada | `data/prepared/`: `{train,val,test}.parquet` y `split_assignment.csv` | salida `data/prepared` en `dvc.lock` |

Los SHA-256 de ambos CSV coinciden con las huellas del anexo del enunciado. Para volver a
descargarlos desde la fuente oficial (SharePoint de UNSW):
`python src/data/download.py --only-splits --out data/original`.

## Preparación y particiones

`src/data/prepare.py` (parámetros en `params.yaml`, sección `data`):

- **Entradas:** 42 variables (3 categóricas: `proto`, `service`, `state`). Se excluyen `id`
  (identificador), `attack_cat` (revela la respuesta) y `label` (objetivo).
- **Limpieza sin estado**, igual para todas las particiones: categóricas sin espacios y
  `is_ftp_login` acotada a {0, 1}. Codificación, escalado e imputación van dentro del
  `Pipeline` del modelo y se ajustan solo con train.
- **Duplicados:** el 42,4 % de las filas de entrenamiento repiten un vector de entradas. Se
  conservan y se agrupan (`group_id` = SHA-1 de las 42 entradas).
- **Contradicciones:** 229 grupos (940 filas) de entrenamiento tienen `label` 0 y 1 a la vez y
  se excluyen de train/val.
- **Train/val:** solo del archivo de entrenamiento, con `StratifiedGroupKFold` (5 folds,
  estratificado por `attack_cat`, semilla 42; el fold 0 es validación). Ningún grupo queda
  repartido entre ambos.
- **Test:** el archivo de test completo, sin eliminar filas. Se reserva para la evaluación final.
- `row_id` = `train:<id>` o `test:<id>` (el `id` del CSV es su número de fila, 1..N).

| Partición | Filas | % ataques |
|---|---|---|
| train | 139.520 | 68,1 |
| val | 34.881 | 68,1 |
| test | 82.332 | 55,1 |
| excluidas (contradictorias) | 940 | |

El 8,1 % de las filas de test comparte vector de entradas con train y el 2,0 % con val. Se
reporta en la auditoría y no se eliminan.

`evidencias/auditoria.json` (etapa `audit`) registra fuente, fecha de descarga, SHA-256, filas,
columnas, tipos, clases, faltantes, duplicados, contradicciones, política de limpieza, variables
usadas y excluidas, semilla, tamaños y huellas de las particiones, y cruces entre ellas.

## Experimentos (MLflow)

Cuatro configuraciones de dos familias, con las mismas particiones (`configs/experiments/`):

| ID | Familia | Cambio | Umbral | Recall | Precision | FPR | Macro-F1 |
|---|---|---|---|---|---|---|---|
| exp01 | Regresión logística | Referencia: one-hot + estandarización | 0,773 | 0,896 | 0,972 | 5,4 % | 0,902 |
| exp02 | Regresión logística | + `log1p` en numéricas | 0,774 | 0,913 | 0,975 | 5,1 % | 0,916 |
| exp03 | Árboles | Random Forest, categóricas ordinales | 0,648 | 0,953 | 0,976 | 5,1 % | 0,945 |
| exp04 | Árboles | HistGradientBoosting | 0,639 | 0,956 | 0,976 | 5,0 % | 0,948 |

Métricas de validación con ataque (1) como clase positiva. Hipótesis, run ID, commit,
referencias DVC, matriz de confusión, métricas con umbral 0,5 y conclusiones en
`evidencias/experimentos.csv`; detección por categoría en `evidencias/errores_por_categoria.csv`.

**Umbral.** Cada configuración elige su umbral con predicciones *out-of-fold* sobre train
(`StratifiedGroupKFold`, 5 folds, mismos grupos que el split): máximo recall de ataque con
FPR ≤ 5 % (`params.yaml`, sección `train`). Validación no interviene en el umbral y queda
limpia para comparar. Regla de decisión: ataque si `score_attack >= umbral`.

**Denominadores cero.** Si una razón tiene denominador 0 (por ejemplo precision sin
predicciones positivas), vale 0,0 y su nombre queda en `undefined` dentro de las métricas.

### Ejecutar un experimento

```bash
git status --porcelain                    # debe estar vacío: el run registra el commit
python -m src.models.train --config configs/experiments/exp04_hist_gradient_boosting.yaml
python -m src.models.export_mlflow        # evidencias/experimentos.csv y evidencias/mlflow/
mlflow ui --backend-store-uri sqlite:///mlflow.db    # opcional: interfaz en http://127.0.0.1:5000
```

El seguimiento es local (`mlflow.db` y `mlruns/`, ignorados por Git). `evidencias/mlflow/`
contiene la exportación de los cuatro runs (`runs.json` con parámetros, métricas y tags, y los
artefactos de cada run) para consultarlos sin el servidor. El modelo de exp03 (52 MB) no se
copia ahí; queda su SHA-256.

Reproducir un experimento da las mismas métricas y predicciones que el run registrado
(comprobado reentrenando exp04 en un clon aparte: diferencia máxima de score 0). Para no mezclar
runs, use un MLflow aparte, por ejemplo `MLFLOW_TRACKING_URI=sqlite:///repro.db`, y no ejecute
`export_mlflow` en un clon nuevo: sobrescribiría `evidencias/` con solo los runs locales.

Tiempos medidos (4 CPU): `dvc repro` de los datos ~2,5 min; cada experimento, entre 16 s
(exp04) y 37 s (exp03), contando las predicciones out-of-fold para el umbral.

## Modelos en Hugging Face (v1 y v2)

Repositorio público `yeico512/unsw-nb15-defiende-el-servidor`. Cada revisión contiene
`model.skops` (Pipeline completo de scikit-learn), `model_config.json` (variables, umbral,
regla de decisión y referencias a Git, DVC y MLflow), `val_metrics.json`, `requirements.txt` y
el model card (`README.md`).

| Versión | Configuración | Revisión (SHA completo) | Modelo SHA-256 | Umbral | Run MLflow |
|---|---|---|---|---|---|
| v1 | `exp01_logreg_base` | `896ee3984613e60d2af1c5297163524216fec34b` | `feb7d024b424fc072332e603ac394024bd7f297f2a2b2ab68b5630a2503d1771` | 0,773 | `b1c856bd8de8457a94df5e64d12c0d58` |
| v2 | `exp04_hist_gradient_boosting` | `424eef599abdce7630cfc369505487755b9763f4` | `562ba1bca9c762c4789a7dcdce4064508d7dc63b180e40b00e57eb302503489c` | 0,639 | `98001a529d484e79a7f6b0cf95dd962d` |

v2 se eligió solo con validación (regla en `configs/seleccion.yaml`: mayor recall con FPR de
validación ≤ 5,5 %) y la decisión quedó congelada en `evidencias/seleccion.json` antes de leer el
test. El model card de v2 recibió después el resultado de test en la revisión
`dd7f3fed3214b7a7d39b20fe067bacb926904c36`, que solo cambia `README.md`; el tag `v2` sigue en la
revisión evaluada.

### Descargar un modelo y predecir

```bash
# v1
python -m src.models.predict --repo-id yeico512/unsw-nb15-defiende-el-servidor \
  --revision 896ee3984613e60d2af1c5297163524216fec34b \
  --input evidencias/recuperacion/entrada.csv --output preds_v1.csv
# v2
python -m src.models.predict --repo-id yeico512/unsw-nb15-defiende-el-servidor \
  --revision 424eef599abdce7630cfc369505487755b9763f4 \
  --input evidencias/recuperacion/entrada.csv --output preds_v2.csv
```

`--input` acepta `.csv` o `.parquet` con las 42 variables (y `row_id` opcional), por ejemplo
`data/prepared/val.parquet`. La salida tiene `row_id`, `score_attack` (probabilidad de ataque) e
`y_pred` (1 si `score_attack >= umbral`). `predict` comprueba que el SHA-256 del modelo
descargado coincide con `model_config.json` y solo confía en los tipos de scikit-learn
listados en `src/models/io.py`.

## Evaluación final en test

v2 congelada, descargada por SHA y evaluada una sola vez sobre las 82.332 filas de test:

| Métrica | Validación | Test |
|---|---|---|
| Recall ataque | 0,956 | 0,968 |
| Precision ataque | 0,976 | 0,867 |
| FPR | 5,0 % | 18,2 % |
| Macro-F1 | 0,948 | 0,898 |

La FPR sube porque en test el 51,8 % del tráfico normal tiene `sttl` 62 o 254 (valores típicos de
los equipos atacantes del banco de pruebas), frente al 23 % en train y validación; todos los
falsos positivos están en ese grupo (`evidencias/analisis-test.json`). El resultado se conserva
sin cambiar modelo ni umbral.

```bash
python -m src.models.recompute_metrics    # recalcula las métricas desde evidencias/predicciones-test.csv
```

## Prueba de recuperación

```bash
export DAGSHUB_USER=<usuario-dagshub> DAGSHUB_TOKEN=<token>
bash scripts/recuperacion.sh /tmp/recuperacion-nueva 91f71a216abc7802ca92faf7b9ac22c317e11b10
```

Clona ese commit en una carpeta nueva, instala el lockfile en un entorno nuevo, recupera los
datos con DVC desde DagsHub (verificando huellas), descarga v1 y v2 por SHA en una caché vacía,
predice sobre 30 filas fijas de validación y compara con las salidas guardadas. Resultado,
log y hashes en `evidencias/recuperacion/` (~5,5 min; resultado: coincidencia exacta).

## Flujo completo (referencia)

```bash
dvc repro                                         # prepare + audit
for c in configs/experiments/*.yaml; do python -m src.models.train --config "$c"; done
python -m src.models.export_mlflow                # experimentos.csv y evidencias/mlflow/
python -m src.models.publish_hf --version v1 --config-id exp01_logreg_base
python -m src.models.select                       # aplica la regla de selección de v2
python -m src.models.publish_hf --version v2 --config-id exp04_hist_gradient_boosting
python -m src.models.select --freeze              # congela la selección antes del test
python -m src.models.evaluate_test                # una sola vez
python -m src.models.analyze_test
python -m src.models.publish_hf --version v2 --config-id exp04_hist_gradient_boosting --update-card
```

Publicar requiere `HF_TOKEN` con permiso de escritura y un repositorio propio (`params.yaml`,
sección `hub`). No repita la publicación ni la evaluación sobre este repositorio: crearían
revisiones nuevas.

## Exploración complementaria

`reports/eda_raw/` contiene un EDA de los 2,54 M registros crudos (`UNSW-NB15_{1..4}.csv`),
hecho antes de fijar el alcance del taller. No forma parte del pipeline: requiere descargar los
crudos (`python src/data/download.py`, ~690 MB) y luego `python -m src.eda.raw_eda`.
