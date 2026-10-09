# Taller MLOps — Detección de intrusiones con UNSW-NB15

Clasificación binaria de tráfico de red (normal vs. ataque) sobre el dataset
[UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset), con
Git (código), DVC (datos y modelos), MLflow (experimentos) y Prefect (pipeline).

## Estado

- [x] Descarga reproducible de los datos (`src/data/download.py`)
- [x] Entorno Python y dependencias
- [x] Datos crudos versionados con DVC (`data/raw.dvc`)
- [x] Exploración de los datos crudos (EDA) y propuesta de split → [`reports/eda_raw/`](reports/eda_raw/README.md)
- [x] Preprocesamiento y splits como pipeline DVC (`dvc.yaml`, `params.yaml`)
- [ ] Experimentos (MLflow)
- [ ] Pipeline (Prefect)

## Reproducir

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt      # versiones exactas (requirements.txt = directas)

python src/data/download.py               # ~690 MB desde el SharePoint de UNSW -> data/raw/
dvc commit data/raw.dvc                   # guarda data/raw en la caché de DVC
git diff --exit-code data/raw.dvc         # sin cambios = los datos coinciden con la versión del repo
dvc repro                                 # EDA -> preprocess -> split (~2.5 min)
```

Aún no hay remoto DVC configurado: las salidas del pipeline se regeneran con `dvc repro`
(es determinista; `dvc.lock` fija los hashes esperados).

`download.py` también escribe `data/raw/MANIFEST.json` con el SHA-256 de cada archivo.
Use `--only-splits` para bajar solo los splits oficiales de entrenamiento/prueba (~47 MB).

## Datos (`data/raw/`)

| Archivo | Contenido |
|---|---|
| `UNSW-NB15_{1..4}.csv` | ~2,54 M registros crudos con timestamps, **sin encabezado** |
| `NUSW-NB15_features.csv` | Nombres y descripción de las 49 columnas de los crudos |
| `UNSW_NB15_training-set.csv` / `UNSW_NB15_testing-set.csv` | Splits oficiales (175.341 / 82.332 filas) |
| `NUSW-NB15_GT.csv`, `UNSW-NB15_LIST_EVENTS.csv` | Ground truth y categorías de ataque |
| `The UNSW-NB15 description.pdf` | Descripción oficial |

## Pipeline DVC

| Etapa | Comando | Salidas |
|---|---|---|
| `eda_raw` | `python -m src.eda.raw_eda` | `reports/eda_raw/` (en git) |
| `preprocess` | `python -m src.data.preprocess` | `data/processed/unsw_nb15.parquet`, `reports/data/preprocess.json` |
| `split` | `python -m src.data.split` | `data/splits/{temporal,holdout}/{train,val,test}.parquet`, `reports/data/splits.json` |

`dvc metrics show` muestra las métricas de datos; los parámetros están en `params.yaml`.

**Preprocesamiento** (`src/data/clean.py` + `src/data/preprocess.py`). Corrige puertos
hexadecimales, `ct_ftp_cmd` en blanco y nombres de `attack_cat`. Imputa con 0 los nulos
FTP/HTTP de la sesión 2 y acota `is_ftp_login` a {0, 1}. Descarta IPs, puertos y timestamps
como features y usa los nombres de columna de los splits oficiales (`smean`, `sload`, `label`…).
Elimina duplicados (mismas features y misma etiqueta; se conserva el registro más antiguo) y
los vectores con etiquetas contradictorias. Solo aplica transformaciones sin estado: la
codificación de categóricas, el escalado, etc. van en el pipeline del modelo y se ajustan
solo con train.

Columnas de salida: 41 features, `attack_cat`, `label` y metadatos `session` y `stime`, que
**no** deben usarse como features.

| | Filas | % ataques |
|---|---|---|
| Crudos | 2.540.047 | 12,65 |
| − duplicados (features + label) | −523.411 | |
| − vectores con etiqueta contradictoria | −926 | |
| **Procesado** | **2.015.710** | **3,86** |

La deduplicación elimina sobre todo ataques repetidos: `Generic` pasa de 215.481 a 17.302
registros. Las métricas sin deduplicar estarían dominadas por copias de unos pocos flujos.

**Splits** (estratificados por `attack_cat`, semilla 42):

| Esquema | train | val | test | % ataques train → test |
|---|---|---|---|---|
| `temporal` (principal): sesión 1 → train/val, sesión 2 → test | 811.852 | 202.963 | 1.000.895 | 1,16 → 6,60 |
| `holdout` (referencia): 70/15/15 sobre ambas sesiones | 1.410.996 | 302.357 | 302.357 | 3,86 → 3,86 |

Tras deduplicar, ningún vector de features aparece en más de una partición. En el split
temporal, train tiene solo 19 `Worms` y unos 9.400 ataques en total. Conviene evaluar con
PR-AUC y recall por categoría.

## EDA de los crudos

```bash
python -m src.eda.raw_eda                 # ~1.5 min, requiere data/raw completo (o: dvc repro eda_raw)
```

Genera [`reports/eda_raw/README.md`](reports/eda_raw/README.md) con tablas (`tables/*.csv`) y
figuras (`figures/*.png`): calidad de datos, distribución del objetivo, comportamiento temporal,
variables categóricas, distribuciones, correlaciones y recomendaciones de preprocesamiento y split.
