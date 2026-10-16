# Taller MLOps — Detección de intrusiones con UNSW-NB15

Clasificación binaria de tráfico de red (normal vs. ataque) sobre el dataset
[UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset), con
Git (código), DVC (datos y modelos), MLflow (experimentos) y Prefect (pipeline).

## Estado

- [x] Descarga reproducible de los datos (`src/data/download.py`)
- [x] Entorno Python y dependencias
- [x] Datos crudos versionados con DVC (`data/raw.dvc`)
- [ ] Exploración y estrategia de split
- [ ] Experimentos (MLflow)
- [ ] Pipeline (Prefect)

## Reproducir

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt      # versiones exactas (requirements.txt = directas)

python src/data/download.py               # ~690 MB desde el SharePoint de UNSW -> data/raw/
dvc status                                # verifica que data/raw coincide con data/raw.dvc
```

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
