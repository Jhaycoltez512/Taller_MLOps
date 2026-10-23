# Defiende el servidor — UNSW-NB15

Taller evaluable de MLOps (Universidad Externado de Colombia, 2026-II): clasificación binaria de
conexiones de red (`label`: 0 = normal, 1 = ataque, clase positiva) con los splits oficiales de
[UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset), versionando código (Git/GitHub),
datos (DVC + DagsHub), experimentos (MLflow) y modelos (Hugging Face Hub).

## Estado

- [x] Datos originales versionados con DVC (`data/original.dvc`) y almacenados en DagsHub
- [x] Auditoría, limpieza y particiones fijas (`dvc.yaml` → `evidencias/auditoria.json`)
- [ ] Cuatro configuraciones de dos familias en MLflow
- [ ] Modelos v1/v2 en Hugging Face Hub
- [ ] Selección congelada y evaluación única en test
- [ ] Prueba de recuperación en entorno nuevo
- [ ] Informe y `entrega.md`

## Requisitos

- Python 3.13 (probado con 3.13.16).
- Dependencias fijadas en `requirements.lock.txt` (`requirements.txt` lista solo las directas).

## Instalación

```bash
git clone https://github.com/Jhaycoltez512/Taller_MLOps.git
cd Taller_MLOps
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt
```

En Debian/Ubuntu, instale dentro del entorno virtual: con el `pip` del sistema la
dependencia `antlr4-python3-runtime` de DVC no compila.

## Recuperar los datos (DVC + DagsHub)

El almacenamiento DVC es `https://dagshub.com/Jhaycoltez512/Taller_MLOps.dvc` (configurado en
`.dvc/config`). Con un usuario de DagsHub que tenga acceso de lectura al repositorio:

```bash
dvc remote modify --local origin user <usuario-dagshub>
dvc remote modify --local origin password <token-dagshub>   # DagsHub → User settings → Tokens
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

## Exploración complementaria

`reports/eda_raw/` contiene un EDA de los 2,54 M registros crudos (`UNSW-NB15_{1..4}.csv`),
hecho antes de fijar el alcance del taller. No forma parte del pipeline: requiere descargar los
crudos (`python src/data/download.py`, ~690 MB) y luego `python -m src.eda.raw_eda`.
