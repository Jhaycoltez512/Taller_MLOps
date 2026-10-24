# Prueba de recuperación

Ejecutada el 2026-10-23 en una carpeta y un entorno nuevos (mismo computador, contenedor Linux,
Python 3.13.16), sin reutilizar cachés de la ejecución original. Sin `HF_TOKEN`: los modelos se
descargaron de forma anónima desde el repositorio público de Hugging Face.

```bash
export DAGSHUB_USER=<usuario> DAGSHUB_TOKEN=<token>
bash scripts/recuperacion.sh <carpeta-nueva> 91f71a216abc7802ca92faf7b9ac22c317e11b10
```

| Paso | Resultado |
|---|---|
| 1. Código | Clon de GitHub en el commit `91f71a216abc7802ca92faf7b9ac22c317e11b10`; `requirements.lock.txt` instalado en un venv nuevo con `pip --no-cache-dir` (298 s) |
| 2. Datos | `dvc pull` desde DagsHub con la caché DVC vacía (9 archivos, 10 s); SHA-256 de los CSV originales = anexo y auditoría; SHA-256 de `train/val/test.parquet` = auditoría |
| 3. v1 | `yeico512/unsw-nb15-defiende-el-servidor@896ee3984613e60d2af1c5297163524216fec34b`, modelo SHA-256 `feb7d024…` |
| 4. v2 | `yeico512/unsw-nb15-defiende-el-servidor@424eef599abdce7630cfc369505487755b9763f4`, modelo SHA-256 `562ba1bc…` |
| 5. Comparación | 30 filas fijas de validación; mismas etiquetas y diferencia máxima de `score_attack` 1,1e-16 en ambas versiones (tolerancia 1e-9): **RECUPERACIÓN OK** |

Tiempo total: 327 s.

Archivos:
- `entrada.csv`: 30 filas fijas de validación (3 por `attack_cat`, semilla 42); `muestra.json`: cómo se eligieron y sus hashes.
- `esperado_v1.csv`, `esperado_v2.csv`: salidas guardadas antes de recuperar (predicciones de validación de los runs originales de MLflow).
- `recuperacion-20261023T205852Z/`: resultado de la ejecución: `log.txt`, `recuperado_v1.csv`, `recuperado_v2.csv`, `comparacion.json`, hashes de datos, modelos y salidas, y `pip-freeze.txt` del entorno nuevo.
